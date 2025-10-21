#include "particle_common.h"  // 包含Particle类定义
#include <SFML/Graphics.hpp>
#include<particle_gpu.cu>
#include <vector>
#include <random>
#include <cstdlib>  // 用于rand()

// 粒子系统（CPU端：管理数据+渲染）
class ParticleSystem {
private:
    // 用Particle类替换原来的ParticleData结构体
    std::vector<Particle> h_particles;  // CPU粒子数据（包含位置、速度等）
    std::vector<sf::Color> colors;      // 仅CPU端使用的颜色数据
    ParticleGPU gpu;                    // GPU计算实例
    sf::CircleShape m_shape;

public:
    ParticleSystem(unsigned int winW, unsigned int winH):m_shape(0) {
        // 初始化随机数生成器
        std::mt19937 gen(std::random_device{}());
        std::uniform_real_distribution<float> posDist(10.0f, winW - 10.0f);  // 位置范围（避免初始就在边界）
        std::uniform_real_distribution<float> speedDist(-50.0f, 50.0f);     // 速度范围
        std::uniform_real_distribution<float> radiusDist(2.0f, 6.0f);       // 半径范围
        int particleCount=1000;
        h_particles.reserve(particleCount);
        colors.reserve(particleCount);

        // 初始化粒子（使用Particle类的构造函数）
        for (int i = 0; i < particleCount; ++i) {
            // 调用Particle类的构造函数初始化位置、速度、半径
            h_particles.emplace_back(
                posDist(gen),    // x坐标
                posDist(gen),    // y坐标
                speedDist(gen),  // x方向速度
                speedDist(gen),  // y方向速度
                radiusDist(gen)  // 半径
            );

            // 随机生成颜色（仅CPU渲染用）
            std::uniform_int_distribution<int> colorDist(0, 255);
            colors.emplace_back
            (
                colorDist(gen),  // R
                colorDist(gen),  // G
                colorDist(gen)   // B
            );
        }

        // 初始化GPU数据（将CPU端粒子数据传入GPU）
        gpu.init(particleCount, h_particles.data());
    }

    // 更新逻辑：调用GPU计算 → 同步数据 → 渲染
    void updateAndRender(sf::RenderWindow& window, float deltaTime) {
        // 1. 调用GPU核函数更新粒子（传入窗口尺寸用于边界判断）
        gpu.update(
            h_particles.size(),
            deltaTime,
            window.getSize().x,
            window.getSize().y
        );

        // 2. 将GPU计算结果同步回CPU内存
        gpu.syncToCPU(h_particles.data(), h_particles.size());
        // 3. 使用SFML渲染粒子
        for (size_t i = 0; i < h_particles.size(); ++i) {
            const Particle& particle = h_particles[i];  // 访问Particle类对象
            m_shape.setRadius(particle.radius);     // 使用类的成员变量

            // 设置位置（SFML的原点在左上角，需偏移半径使圆心对齐粒子坐标）
            m_shape.setPosition
            (
                sf::Vector2<float>
                (
                    particle.x - particle.radius,  // x 坐标
                    particle.y - particle.radius   // y 坐标
                )
            );
            m_shape.setFillColor(colors[i]);  // 设置颜色
            window.draw(m_shape);
        }
    }
};

// 主函数：窗口初始化与主循环
int main() {
    // 创建SFML窗口
    sf::RenderWindow window(
    sf::VideoMode(sf::Vector2u(800, 600)),  // 显式传递 Vector2u
    "GPU粒子系统（使用Particle类）",
    sf::Style::Close
);
    window.setFramerateLimit(60);  // 限制帧率为60FPS
    sf::Clock clock;               // 用于计算帧间隔时间

    // 初始化粒子系统
    ParticleSystem particleSystem(window.getSize().x, window.getSize().y);

    // 主循环
    while (window.isOpen()) {
        // 计算帧间隔时间（秒）
        float deltaTime = clock.restart().asSeconds();
        // 事件处理（关闭窗口）
        while (const std::optional<sf::Event> event = window.pollEvent())
        {
            if (event->is<sf::Event::Closed>()) 
            {
                window.close();
            } 
            else if (event->is<sf::Event::KeyPressed>()) 
            {
                if (const auto* keyEvent = event->getIf<sf::Event::KeyPressed>()) 
                {
                    if (keyEvent->code == sf::Keyboard::Key::Delete) 
                    {
                        window.close();
                    }
                }
            } 
            else if (const auto* resizeEvent = event->getIf<sf::Event::Resized>()) 
            {
                sf::Vector2u newSize = resizeEvent->size;
                // 处理窗口大小变化...
                
            }
        }
            // 清空窗口（黑色背景）
            window.clear(sf::Color::Black);

            // 更新并渲染粒子
            particleSystem.updateAndRender(window, deltaTime);

            // 显示当前帧
            window.display();
        }

    return 0;
    }

    