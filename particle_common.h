// 仅包含 CPU/GPU 通用的数据结构（无 SFML、无 CUDA 关键字）
#ifndef PARTICLE_COMMON_H
#define PARTICLE_COMMON_H
#include<iostream>
#include<string>
using namespace std;
// 用 class 替代 struct，存储粒子数据
class Particle {
public:
    // 成员变量（需 public，否则 vector 无法直接访问，或提供访问接口）
    float x, y;       // 位置
    float vx, vy;     // 速度
    float radius;     // 半径
    float *near_particles[10];
    double partticles_distance[999];
    Particle *partticles_distance_idx[999];
    Particle *front_ten[10];

    // 构造函数（可选，方便初始化）
    Particle(float x_ = 0, float y_ = 0, float vx_ = 0, float vy_ = 0, float r_ = 1.0f)
        : x(x_), y(y_), vx(vx_), vy(vy_), radius(r_) {
            for(int i=0;i<10;i++)
            {
                near_particles[i]=nullptr;
            }
        }
};

#endif // PARTICLE_COMMON_H