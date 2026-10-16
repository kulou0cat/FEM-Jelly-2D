#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <Eigen/Dense>             // Eigen 线性代数库：直接替代手写的 Vec2 和 Mat2
#include <opencv2/opencv.hpp>      // OpenCV 视觉库：直接弹窗实时渲染 2D 网格动画

struct TrigangleElement
{
	int a, b, c;
	Eigen::Matrix2d B;
	double V_rest;
	double last_J=1.0;
};

class JellySimulator2D {
public:
	double E = 14000.0;
	double nu = 0.35;
	double mu = E / (2.0 * (1.0 + nu));
	double lamba = (E * nu) / ((1.0 + nu) * (1.0 - 2.0 * nu));
	double density = 1000.0;
	double damping = 1.2;
	Eigen::Vector2d gravity{ 0.0,-9.8 };

	std::vector<Eigen::Vector2d> x, v, f;
	std::vector<double> mass;
	std::vector<TrigangleElement> elements;

	JellySimulator2D(int grid_n = 4) {
		int nv = grid_n + 1;
		double dx = 0.34 / grid_n;
		Eigen::Rotation2Dd rot(0.25);
		Eigen::Vector2d center(0.5, 0.55);

		x.resize(nv * nv);
		v.assign(nv * nv, Eigen::Vector2d::Zero());
		f.assign(nv * nv, Eigen::Vector2d::Zero());
		mass.assign(nv * nv, 0.0);

		for (int j = 0; j < nv; ++j) {
			for (int i = 0; i < nv; ++i){
				Eigen::Vector2d local((i - grid_n * 0.5) * dx,(j - grid_n * 0.5) * dx);
				x[j * nv + i] = center + rot * local;
			}
		}

		for (int j = 0; j < grid_n; ++j) {
			for (int i = 0; i < grid_n; ++i) {
				int v00 = j * nv + i;
				int v10 = v00 + 1;
				int v01 = (j + 1) * nv + i;
				int v11 = v01 + 1;
				add_triangle(v00, v10, v11);
				add_triangle(v00, v11, v01);
			}
		}
	}

	void add_triangle(int a, int b, int c) {
		Eigen::Matrix2d Dm;//初始边矩阵
		Dm.col(0) = x[a] - x[c];
		Dm.col(1) = x[b] - x[c];
		double V_rest = 0.5 * std::abs(Dm.determinant());
		elements.push_back({ a,b,c,Dm.inverse(),V_rest });
		double m_node = (density * V_rest) / 3.0;
		mass[a] += m_node;
		mass[b] += m_node;
		mass[c] += m_node;
	}

	void substep(double dt) {
		for (size_t i = 0; i < x.size(); ++i) {
			f[i] = mass[i] * (gravity - damping * v[i]);
		}

		for (auto& e : elements) {
			Eigen::Matrix2d Ds;//当前边矩阵
			Ds.col(0) = x[e.a] - x[e.c];
			Ds.col(1) = x[e.b] - x[e.c];

			Eigen::Matrix2d F = Ds * e.B;
			double J = std::max(0.15, F.determinant());
			e.last_J = J;

			Eigen::Matrix2d F_inv_T = F.inverse().transpose();
			Eigen::Matrix2d P = mu * (F - F_inv_T) + lamba * std::log(J) * F_inv_T;
			Eigen::Matrix2d H = -e.V_rest * P * e.B.transpose();
			f[e.a] += H.col(0);
			f[e.b] += H.col(1);
			f[e.c] -=(H.col(0)+H.col(1));
		}

		for (size_t i = 0; i < x.size(); ++i) {
			v[i] += ((f[i] / mass[i]) * dt);
			x[i] += (v[i] * dt);
			if (x[i].y() < 0.08) {
				x[i].y() = 0.08;
				if (v[i].y() < 0.0) {
					v[i].y() *= -0.55;
					v[i].x() *= 0.90;
				}
			}
			if (x[i].x() < 0.05 ) {
				x[i].x() = 0.05;
				v[i].x() = std::abs(v[i].x()) * 0.5;
			}
			if (x[i].x() >0.95) {
				x[i].x() = 0.95;
				v[i].x() =- std::abs(v[i].x()) * 0.5;
			}
		}
	}

};

int main() {
    JellySimulator2D sim(4);

    const int W = 600;
    const int H = 600;
    const char* window_name = "2D FEM Jelly - Mouse Drag";

    cv::Mat canvas(H, W, CV_8UC3);

    // 固定图像大小，方便将鼠标坐标转换成物理坐标
    cv::namedWindow(window_name, cv::WINDOW_AUTOSIZE);

    // 保存鼠标拖动所需的数据
    struct MouseState {
        JellySimulator2D* sim = nullptr;

        int width = 600;
        int height = 600;

        // -1 表示没有抓住任何顶点
        int selected = -1;

        // 鼠标位置对应的物理坐标
        Eigen::Vector2d target = Eigen::Vector2d::Zero();
    };

    MouseState mouse;
    mouse.sim = &sim;
    mouse.width = W;
    mouse.height = H;

    // 注册鼠标回调函数：
    // 鼠标点击、移动、松开时，OpenCV 会调用这个函数
    cv::setMouseCallback(
        window_name,
        [](int event, int mx, int my, int flags, void* userdata) {
            MouseState& state =
                *static_cast<MouseState*>(userdata);

            // 将像素坐标转换成物理坐标
            double world_x =
                static_cast<double>(mx) / (state.width - 1);

            double world_y =
                1.0 - static_cast<double>(my) / (state.height - 1);

            // 将拖动目标限制在模拟区域内
            state.target = Eigen::Vector2d(
                std::clamp(world_x, 0.05, 0.95),
                std::clamp(world_y, 0.08, 0.95)
            );

            // 按下左键：寻找鼠标附近最近的顶点
            if (event == cv::EVENT_LBUTTONDOWN) {
                state.selected = -1;

                // 最远允许抓取距离：35 像素
                const double pick_radius = 35.0;
                double best_distance2 = pick_radius * pick_radius;

                for (int i = 0;
                    i < static_cast<int>(state.sim->x.size());
                    ++i) {

                    const Eigen::Vector2d& p = state.sim->x[i];

                    // 顶点的像素坐标
                    double px = p.x() * (state.width - 1);
                    double py = (1.0 - p.y()) * (state.height - 1);

                    double dx = px - mx;
                    double dy = py - my;

                    double distance2 = dx * dx + dy * dy;

                    if (distance2 < best_distance2) {
                        best_distance2 = distance2;
                        state.selected = i;
                    }
                }
            }

            // 松开左键：释放顶点
            if (event == cv::EVENT_LBUTTONUP) {
                state.selected = -1;
            }

            // 避免在窗口外松开后，回到窗口仍保持抓取
            if (event == cv::EVENT_MOUSEMOVE &&
                (flags & cv::EVENT_FLAG_LBUTTON) == 0) {
                state.selected = -1;
            }
        },
        &mouse
    );

    // 物理坐标 → 像素坐标
    auto to_cv_pt = [&](const Eigen::Vector2d& p) {
        return cv::Point(
            static_cast<int>(p.x() * (W - 1)),
            static_cast<int>((1.0 - p.y()) * (H - 1))
        );
        };

    const double dt = 1e-4;
    const int substeps = 150;

    // 鼠标弹簧参数：采用除以质量后的系数
    const double drag_stiffness = 800.0;
    const double drag_damping = 45.0;

    while (true) {
        // ---------- 物理模拟 ----------
        for (int s = 0; s < substeps; ++s) {

            if (mouse.selected >= 0) {
                int i = mouse.selected;

                // 从被抓顶点指向鼠标目标的向量
                Eigen::Vector2d displacement =
                    mouse.target - sim.x[i];

                // 限制弹簧用于计算拉力的伸长量，
                // 减轻鼠标突然移动很远造成的冲击
                const double max_extension = 0.08;
                double length = displacement.norm();

                if (length > max_extension) {
                    displacement *= max_extension / length;
                }

                // 弹簧加速度 + 阻尼加速度
                Eigen::Vector2d drag_acceleration =
                    drag_stiffness * displacement
                    - drag_damping * sim.v[i];

                // 先施加鼠标作用产生的速度变化
                sim.v[i] += drag_acceleration * dt;
            }

            // 再执行原来的重力、弹性力和碰撞更新
            sim.substep(dt);
        }

        // ---------- 绘图 ----------
        canvas.setTo(cv::Scalar(40, 25, 15));

        // 地面
        int gy = static_cast<int>((1.0 - 0.08) * (H - 1));

        cv::line(
            canvas,
            cv::Point(20, gy),
            cv::Point(W - 20, gy),
            cv::Scalar(140, 120, 100),
            3
        );

        // 三角形及边线
        for (const auto& e : sim.elements) {
            cv::Point pts[3] = {
                to_cv_pt(sim.x[e.a]),
                to_cv_pt(sim.x[e.b]),
                to_cv_pt(sim.x[e.c])
            };

            int r = std::clamp(
                static_cast<int>(220 - (e.last_J - 0.85) * 450),
                40, 255
            );

            int b = std::clamp(
                static_cast<int>(60 + (e.last_J - 0.85) * 450),
                40, 255
            );

            cv::fillConvexPoly(
                canvas,
                pts,
                3,
                cv::Scalar(b, 130, r),
                cv::LINE_AA
            );

            // 分别绘制三条边
            for (int k = 0; k < 3; ++k) {
                cv::line(
                    canvas,
                    pts[k],
                    pts[(k + 1) % 3],
                    cv::Scalar(245, 245, 245),
                    1,
                    cv::LINE_AA
                );
            }
        }

        // 画出顶点，方便鼠标选择
        for (const auto& p : sim.x) {
            cv::circle(
                canvas,
                to_cv_pt(p),
                3,
                cv::Scalar(240, 240, 240),
                -1,
                cv::LINE_AA
            );
        }

        // 显示被抓住的顶点和鼠标弹簧
        if (mouse.selected >= 0) {
            cv::Point grabbed = to_cv_pt(sim.x[mouse.selected]);
            cv::Point target = to_cv_pt(mouse.target);

            cv::line(
                canvas,
                grabbed,
                target,
                cv::Scalar(0, 255, 255),
                2,
                cv::LINE_AA
            );

            cv::circle(
                canvas,
                grabbed,
                6,
                cv::Scalar(0, 255, 255),
                -1,
                cv::LINE_AA
            );

            cv::circle(
                canvas,
                target,
                7,
                cv::Scalar(0, 255, 255),
                2,
                cv::LINE_AA
            );
        }

        cv::putText(
            canvas,
            "Hold LMB near a vertex to drag",
            cv::Point(15, 28),
            cv::FONT_HERSHEY_SIMPLEX,
            0.60,
            cv::Scalar(200, 220, 240),
            1,
            cv::LINE_AA
        );

        cv::putText(
            canvas,
            "R: Reset | ESC: Quit",
            cv::Point(15, 53),
            cv::FONT_HERSHEY_SIMPLEX,
            0.60,
            cv::Scalar(200, 220, 240),
            1,
            cv::LINE_AA
        );

        cv::imshow(window_name, canvas);

        // 处理窗口、鼠标和键盘事件
        int key = cv::waitKey(15);

        if (key == 27) {
            break;
        }

        if (key == 'r' || key == 'R') {
            mouse.selected = -1;
            sim = JellySimulator2D(4);
        }

        // 点击窗口右上角关闭按钮时退出
        if (cv::getWindowProperty(
            window_name, cv::WND_PROP_VISIBLE) < 1.0) {
            break;
        }
    }

    cv::destroyAllWindows();
    return 0;
}