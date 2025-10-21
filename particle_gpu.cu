#include "particle_common.h"
#include <cuda_runtime.h>
#define CUDACheck(expr) do { \
    cudaError_t err = (expr); \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA 错误: " << cudaGetErrorString(err) \
                  << " (行号: " << __LINE__ << ", 文件: " << __FILE__ << ")" << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while (0)

// GPU 核函数：更新粒子位置（力场计算逻辑）
__global__ void updateParticles(Particle* d_particles, int count, float deltaTime, 
                               unsigned int winWidth, unsigned int winHeight) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;//线程的全局索引
    if (idx >= count) return;

    //Particle& p = d_particles[idx];
    // 示例：简单运动+边界反弹（后续替换为力场逻辑)
    extern __shared__ Particle share_particles[];
    int tid = threadIdx.x;
    if(tid>=blockDim.x)
    {
        return;
    }
    share_particles[tid]=d_particles[idx];
    share_particles[tid].x += share_particles[tid].vx * deltaTime;
    share_particles[tid].y += share_particles[tid].vy * deltaTime;

    if (share_particles[tid].x - share_particles[tid].radius < 0 || share_particles[tid].x + share_particles[tid].radius > winWidth) {
        share_particles[tid].vx *= -0.8f;
        share_particles[tid].x = max(share_particles[tid].radius, min(share_particles[tid].x, (float)winWidth - share_particles[tid].radius));
    }
    if (share_particles[tid].y - share_particles[tid].radius < 0 || share_particles[tid].y + share_particles[tid].radius > winHeight) {
        share_particles[tid].vy *= -0.8f;
        share_particles[tid].y = max(share_particles[tid].radius, min(share_particles[tid].y, (float)winHeight - share_particles[tid].radius));
    }
}
__global__ void compute_distance(Particle* d_particles,int m_x,int m_y,int m_index,int count)//在初始所有粒子之后进行粒子距离运算
{  
    int idx = blockIdx.x * blockDim.x + threadIdx.x;//线程的全局索引
    if (idx >= count) return;
    //Particle& p = d_particles[idx];//保存在本地内存，采用将全局内存加载到共享内存上
    extern __shared__ Particle share_particles[];
    int tid = threadIdx.x;
    if(tid<blockDim.x)
    {
        share_particles[tid]=d_particles[idx];
    }
    
    if(m_index==idx)
    {
        return;//除去自己以外的任何粒子
    }

    else{
        double distance=(m_x-share_particles[tid].x)*(m_x-share_particles[tid].x)+(m_y-share_particles[tid].y)*(m_y-share_particles[tid].y);
        share_particles[tid].partticles_distance[idx]=distance;//获取其余粒子对某个粒子的距离放在该列表
        share_particles[tid].partticles_distance_idx[idx]=&d_particles[idx];//获取其余粒子的索引
    }
}
__global__ void update_near_particles(Particle* d_particles, int count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;//线程的全局索引
    if (idx >= count) return;
    //Particle& p = d_particles[idx];//保存在本地内存，采用将全局内存加载到共享内存上

    extern __shared__ Particle share_particles[];
    int tid = threadIdx.x;
    if(tid<blockDim.x)
    {
        share_particles[tid]=d_particles[idx];
    }

    int computer_counts=count;
    dim3 block(256);
    dim3 grid((computer_counts + 256 - 1) / 256);
    compute_distance<<<block,grid>>>(&share_particles[tid],share_particles[tid].x,share_particles[tid].y,idx,count);
    for (int i = 0; i < count - 1; i++) {
        // 每轮循环将最大元素"浮"到末尾
        for (int j = 0; j < count - i - 1; j++) {
            if (share_particles[tid].partticles_distance[j] > share_particles[tid].partticles_distance[j + 1]) {
                // 交换元素
                int temp=share_particles[tid].partticles_distance[j];
                Particle *temp_idx=share_particles[tid].partticles_distance_idx[j];
                share_particles[tid].partticles_distance[j]=share_particles[tid].partticles_distance[j+1];
                share_particles[tid].partticles_distance_idx[j]=share_particles[tid].partticles_distance_idx[j+1];
                share_particles[tid].partticles_distance[j+1]=temp;
                share_particles[tid].partticles_distance_idx[j+1]=temp_idx;//把索引以及距离值交换
            }
        }
    }
    
}

// 封装 CUDA 操作的类（供 CPU 调用）
class ParticleGPU {
private:
    Particle* d_particles = nullptr; // GPU 内存指针
public:
    // 分配 GPU 内存
    void init(int count, const Particle* h_particles) {
        cudaMalloc(&d_particles, count * sizeof(Particle));
        cudaMemcpy(d_particles, h_particles, count * sizeof(Particle), cudaMemcpyHostToDevice);

    }

    // 调用 GPU 核函数更新粒子
    void update(int count, float deltaTime, unsigned int winW, unsigned int winH) 
    {
        if (count <= 0) return;  // 避免无效计算
        
        dim3 block(256);
        dim3 grid((count + block.x - 1) / block.x);  // 计算网格大小
        
        // 启动第一个内核：更新粒子力学状态
        updateParticles<<<grid, block>>>(d_particles, count, deltaTime, winW, winH);
        CUDACheck(cudaGetLastError());  // 检查内核启动错误
        CUDACheck(cudaDeviceSynchronize());  // 同步设备，确保内核执行完成（可选，根据需求）
        
        // 启动第二个内核：计算邻近粒子（修正块大小参数和函数名）
        update_near_particles<<<grid, block>>>(d_particles, count);  // 第二个参数必须是 blockSize
        CUDACheck(cudaGetLastError());  // 检查内核启动错误
        CUDACheck(cudaDeviceSynchronize());  // 同步设备
    }

        // 将 GPU 数据同步回 CPU
    void syncToCPU(Particle* h_particles, int count) 
    {
        cudaMemcpy(h_particles, d_particles, count * sizeof(Particle), cudaMemcpyDeviceToHost);
        cudaGetLastError();
    }

    // 释放 GPU 内存
    ~ParticleGPU() {
        if (d_particles) cudaFree(d_particles);
    }
};