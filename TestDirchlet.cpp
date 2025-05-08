#include <iostream>
#include <iomanip>

#include <cmath>

#include <string>
#include <vector>
#include <array>
#include <algorithm>

#include <limits>
#include <numeric>

#include <random>
#include <complex>
#include <bitset>

#include <bit>

//https://zh.wikipedia.org/wiki/%E6%B7%B7%E6%B2%8C%E7%90%86%E8%AE%BA
//https://en.wikipedia.org/wiki/Chaos_theory
namespace ChaoticTheory
{
	//模拟双段摆锤物理系统，根据二进制密钥生成伪随机实数
	//Simulate a two-segment pendulum physical system to generate pseudo-random real numbers based on a binary key
	//Reference:
	//https://zh.wikipedia.org/wiki/%E5%8F%8C%E6%91%86
	//https://en.wikipedia.org/wiki/Double_pendulum
	//https://www.myphysicslab.com/pendulum/double-pendulum-en.html
	//https://www.researchgate.net/publication/345243089_A_Pseudo-Random_Number_Generator_Using_Double_Pendulum
	//https://github.com/robinsandhu/DoublePendulumPRNG/blob/master/prng.cpp
	class SimulateDoublePendulum
	{

	private:

		//备份：θ1, θ2
		std::array<long double, 2> BackupTensions{};
		//备份：ω1, ω2
		std::array<long double, 2> BackupVelocitys{};

		/* SystemData 索引表
		 *  0  RodLengthUpper   (L1)
		 *  1  RodLengthLower   (L2)
		 *  2  MassUpper        (m1)
		 *  3  MassLower        (m2)
		 *  4  AngleUpper       (θ1)
		 *  5  AngleLower       (θ2)
		 *  6  InitializationRadius  (extra seed)
		 *  7  KeySize          (for initialization)
		 *  8  AngularVelocityUpper (ω1)
		 *  9  AngularVelocityLower (ω2)
		 */
		std::array<long double, 10> SystemData{};

		static constexpr long double gravity_coefficient = 9.80665; //Gravity Acceleration (meter/second^{2})
		static constexpr long double hight = 0.002; //time step uint (second)

		// -----------------------------------------------------------------------------
		//  Four-Order Runge–Kutta integrator for one time-step
		// -----------------------------------------------------------------------------
		inline void rk4_single_step(long double timeStep)
		{
			long double& RodLengthUpper = this->SystemData[0];
			long double& RodLengthLower = this->SystemData[1];
			long double& MassUpper = this->SystemData[2];
			long double& MassLower = this->SystemData[3];
			long double& AngleUpper = this->SystemData[4];
			long double& AngleLower = this->SystemData[5];
			long double& AngularVelocityUpper = this->SystemData[8];
			long double& AngularVelocityLower = this->SystemData[9];

			// ----------- 便捷 Lambda：计算加速度 -------------
			auto calculate_acceleration = [&](long double theta1, long double theta2,
				long double omega1, long double omega2,
				long double& alpha1, long double& alpha2)
				{
					long double deltaAngle = theta1 - theta2;
					long double commonDenominator = (2 * MassUpper + MassLower)
						- MassLower * std::cos(2 * deltaAngle);
					long double denominator1 = RodLengthUpper * commonDenominator;
					long double denominator2 = RodLengthLower * commonDenominator;

					// α₁
					alpha1 = -gravity_coefficient * (2 * MassUpper + MassLower) * std::sin(theta1)
						- MassLower * gravity_coefficient * std::sin(theta1 - 2 * theta2)
						- 2 * std::sin(deltaAngle) * MassLower *
						(omega2 * omega2 * RodLengthLower
							+ omega1 * omega1 * RodLengthUpper * std::cos(deltaAngle));
					alpha1 /= denominator1;

					// α₂
					alpha2 = 2 * std::sin(deltaAngle) *
						(omega1 * omega1 * RodLengthUpper * (MassUpper + MassLower)
							+ gravity_coefficient * (MassUpper + MassLower) * std::cos(theta1)
							+ omega2 * omega2 * RodLengthLower * MassLower * std::cos(deltaAngle));
					alpha2 /= denominator2;
				};

			// ---------------- RK‑4 计算 ---------------------
			//k1
			long double k1_theta1 = AngularVelocityUpper;
			long double k1_theta2 = AngularVelocityLower;
			long double k1_omega1, k1_omega2;
			calculate_acceleration(AngleUpper, AngleLower,
				AngularVelocityUpper, AngularVelocityLower,
				k1_omega1, k1_omega2);

			//k2
			long double theta1_mid = AngleUpper + 0.5L * timeStep * k1_theta1;
			long double theta2_mid = AngleLower + 0.5L * timeStep * k1_theta2;
			long double omega1_mid = AngularVelocityUpper + 0.5L * timeStep * k1_omega1;
			long double omega2_mid = AngularVelocityLower + 0.5L * timeStep * k1_omega2;
			long double k2_theta1 = omega1_mid;
			long double k2_theta2 = omega2_mid;
			long double k2_omega1, k2_omega2;
			calculate_acceleration(theta1_mid, theta2_mid, omega1_mid, omega2_mid,
				k2_omega1, k2_omega2);

			//k3
			theta1_mid = AngleUpper + 0.5L * timeStep * k2_theta1;
			theta2_mid = AngleLower + 0.5L * timeStep * k2_theta2;
			omega1_mid = AngularVelocityUpper + 0.5L * timeStep * k2_omega1;
			omega2_mid = AngularVelocityLower + 0.5L * timeStep * k2_omega2;
			long double k3_theta1 = omega1_mid;
			long double k3_theta2 = omega2_mid;
			long double k3_omega1, k3_omega2;
			calculate_acceleration(theta1_mid, theta2_mid, omega1_mid, omega2_mid,
				k3_omega1, k3_omega2);

			//k4
			long double theta1_end = AngleUpper + timeStep * k3_theta1;
			long double theta2_end = AngleLower + timeStep * k3_theta2;
			long double omega1_end = AngularVelocityUpper + timeStep * k3_omega1;
			long double omega2_end = AngularVelocityLower + timeStep * k3_omega2;
			long double k4_theta1 = omega1_end;
			long double k4_theta2 = omega2_end;
			long double k4_omega1, k4_omega2;
			calculate_acceleration(theta1_end, theta2_end, omega1_end, omega2_end,
				k4_omega1, k4_omega2);

			// 更新 state
			AngleUpper += (timeStep / 6.0L) * (k1_theta1 + 2 * k2_theta1 + 2 * k3_theta1 + k4_theta1);
			AngleLower += (timeStep / 6.0L) * (k1_theta2 + 2 * k2_theta2 + 2 * k3_theta2 + k4_theta2);
			AngularVelocityUpper += (timeStep / 6.0L) * (k1_omega1 + 2 * k2_omega1 + 2 * k3_omega1 + k4_omega1);
			AngularVelocityLower += (timeStep / 6.0L) * (k1_omega2 + 2 * k2_omega2 + 2 * k3_omega2 + k4_omega2);
		}

		void run_system_rk4(bool is_initialize_mode, std::uint64_t step_count)
		{
			if (step_count == 0)
				return;

			for (std::uint64_t i = 0; i < step_count; ++i)
			{
				rk4_single_step(hight);
			}

			if (is_initialize_mode)
			{
				this->BackupTensions[0] = this->SystemData[4];
				this->BackupTensions[1] = this->SystemData[5];
				this->BackupVelocitys[0] = this->SystemData[8];
				this->BackupVelocitys[1] = this->SystemData[9];
			}
		}

		void run_system_euler(bool is_initialize_mode, std::uint64_t time)
		{
			const long double& length1 = this->SystemData[0];
			const long double& length2 = this->SystemData[1];
			const long double& mass1 = this->SystemData[2];
			const long double& mass2 = this->SystemData[3];
			long double& tension1 = this->SystemData[4];
			long double& tension2 = this->SystemData[5];

			long double& velocity1 = this->SystemData[8];
			long double& velocity2 = this->SystemData[9];

			for (std::uint64_t counter = 0; counter < time; ++counter)
			{
				long double denominator = 2 * mass1 + mass2 - mass2 * ::cos(2 * tension1 - 2 * tension2);

				long double alpha1 = -1 * gravity_coefficient * (2 * mass1 + mass2) * ::sin(tension1)
					- mass2 * gravity_coefficient * ::sin(tension1 - 2 * tension2)
					- 2 * ::sin(tension1 - tension2) * mass2
					* (velocity2 * velocity2 * length2 + velocity1 * velocity1 * length1 * ::cos(tension1 - tension2));

				alpha1 /= length1 * denominator;

				long double alpha2 = 2 * ::sin(tension1 - tension2)
					* (velocity1 * velocity1 * length1 * (mass1 + mass2) + gravity_coefficient * (mass1 + mass2) * ::cos(tension1) + velocity2 * velocity2 * length2 * mass2 * ::cos(tension1 - tension2));

				alpha2 /= length2 * denominator;

				velocity1 += hight * alpha1;
				velocity2 += hight * alpha2;
				tension1 += hight * velocity1;
				tension2 += hight * velocity2;
			}

			if (is_initialize_mode)
			{
				this->BackupTensions[0] = tension1;
				this->BackupTensions[1] = tension2;

				this->BackupVelocitys[0] = velocity1;
				this->BackupVelocitys[1] = velocity2;
			}
		}

		void initialize(std::vector<std::int8_t>& binary_key_sequence)
		{
			if (binary_key_sequence.empty())
				return;

			const std::size_t binary_key_sequence_size = binary_key_sequence.size();
			std::vector<std::vector<std::int8_t>> binary_key_sequence_2d(4, std::vector<std::int8_t>());
			for (std::size_t index = 0; index < binary_key_sequence_size / 4; index++)
			{
				binary_key_sequence_2d[0].push_back(binary_key_sequence[index]);
				binary_key_sequence_2d[1].push_back(binary_key_sequence[binary_key_sequence_size / 4 + index]);
				binary_key_sequence_2d[2].push_back(binary_key_sequence[binary_key_sequence_size / 2 + index]);
				binary_key_sequence_2d[3].push_back(binary_key_sequence[binary_key_sequence_size * 3 / 4 + index]);
			}

			std::vector<std::vector<std::int8_t>> binary_key_sequence_2d_param(7, std::vector<std::int8_t>());
			std::int32_t key_outer_round_count = 0;
			std::int32_t key_inner_round_count = 0;
			while (key_outer_round_count < 64)
			{
				while (key_inner_round_count < binary_key_sequence_size / 4)
				{
					binary_key_sequence_2d_param[0].push_back(binary_key_sequence_2d[0][key_inner_round_count] ^ binary_key_sequence_2d[1][key_inner_round_count]);
					binary_key_sequence_2d_param[1].push_back(binary_key_sequence_2d[0][key_inner_round_count] ^ binary_key_sequence_2d[2][key_inner_round_count]);
					binary_key_sequence_2d_param[2].push_back(binary_key_sequence_2d[0][key_inner_round_count] ^ binary_key_sequence_2d[3][key_inner_round_count]);
					binary_key_sequence_2d_param[3].push_back(binary_key_sequence_2d[1][key_inner_round_count] ^ binary_key_sequence_2d[2][key_inner_round_count]);
					binary_key_sequence_2d_param[4].push_back(binary_key_sequence_2d[1][key_inner_round_count] ^ binary_key_sequence_2d[3][key_inner_round_count]);
					binary_key_sequence_2d_param[5].push_back(binary_key_sequence_2d[2][key_inner_round_count] ^ binary_key_sequence_2d[3][key_inner_round_count]);
					binary_key_sequence_2d_param[6].push_back(binary_key_sequence_2d[0][key_inner_round_count]);

					++key_inner_round_count;
					++key_outer_round_count;
					if (key_outer_round_count >= 64)
					{
						break;
					}
				}
				key_inner_round_count = 0;
			}
			key_outer_round_count = 0;

			long double& radius = this->SystemData[6];
			long double& current_binary_key_sequence_size = this->SystemData[7];

			for (std::int32_t i = 0; i < 64; i++)
			{
				for (std::int32_t j = 0; j < 6; j++)
				{
					if (binary_key_sequence_2d_param[j][i] == 1)
						this->SystemData[j] += 1 * ::powl(2.0, 0 - i);
				}
				if (binary_key_sequence_2d_param[6][i] == 1)
					radius += 1 * ::powl(2.0, 4 - i);
			}

			current_binary_key_sequence_size = static_cast<long double>(binary_key_sequence_size);

			//This is initialize mode
			this->run_system_rk4(true, static_cast<std::uint64_t>(::round(radius * current_binary_key_sequence_size)));
		}

		long double generate()
		{
			//This is generate mode
			/* 0) 先走一步，获得当前 state */
			this->run_system_rk4(false, 1);

			/* ------ ① 计算动态 scrambleFactor ------ */
			long double scrambleFactorSeed = this->SystemData[4] + this->SystemData[5] + this->SystemData[8] + this->SystemData[9];

			unsigned int e2 = static_cast<unsigned int>(std::log2(scrambleFactorSeed * 1.0e6L)) % 11u;// 0‥10
			unsigned int mantissa = static_cast<unsigned int>(scrambleFactorSeed * 1.0e6L * 997.0L) % 97u;// 0‥96
			unsigned int scrambleFactor = (251u + (mantissa | 1u)) << e2;      // 251‥≈360k

			/* ------ ② 原始幅值组合 ------ */
			long double temporary_floating_a = this->SystemData[0] * ::sin(this->SystemData[4])
				+ this->SystemData[1] * ::sin(this->SystemData[5]);
			long double temporary_floating_b = -this->SystemData[0] * ::sin(this->SystemData[4])
				- this->SystemData[1] * ::sin(this->SystemData[5]);

			/* ------ ③ 动态因子放缩，再 ×2^32 ------ */
			constexpr long double twoPow32 = 4294967296.0L;
			temporary_floating_a = ::fmod(temporary_floating_a * static_cast<long double>(scrambleFactor), 1.0L) * twoPow32;
			temporary_floating_b = ::fmod(temporary_floating_b * static_cast<long double>(scrambleFactor), 1.0L) * twoPow32;

			if (std::isnan(temporary_floating_a) || std::isnan(temporary_floating_b) || std::isinf(temporary_floating_a) || std::isinf(temporary_floating_b))
				return 0.0L;

			/* 再跑一步保持节奏 */
			this->run_system_rk4(false, 1);

			return (temporary_floating_a * 2.0L + temporary_floating_b >= 0.0L) ? temporary_floating_a : temporary_floating_b;
		}

	public:

		using result_type = long double;

		static constexpr result_type min()
		{
			return std::numeric_limits<result_type>::lowest();
		}

		static constexpr result_type max()
		{
			return std::numeric_limits<result_type>::max();
		};

		result_type operator()()
		{
			return this->generate();
		}

		/*──────────── Reset to backup state ────────────*/
		void reset()
		{
			this->SystemData[4] = this->BackupTensions[0];
			this->SystemData[5] = this->BackupTensions[1];
			this->SystemData[8] = this->BackupVelocitys[0];
			this->SystemData[9] = this->BackupVelocitys[1];
		}

		void seed_with_binary_string(std::string binary_key_sequence_string)
		{
			std::vector<int8_t> binary_key_sequence;
			std::string_view view_only_string(binary_key_sequence_string);
			const char binary_zero_string = '0';
			const char binary_one_string = '1';
			for (const char& data : view_only_string)
			{
				if (data != binary_zero_string && data != binary_one_string)
					continue;

				binary_key_sequence.push_back(data == binary_zero_string ? 0 : 1);
			}

			if (binary_key_sequence.empty())
				return;
			else
				this->initialize(binary_key_sequence);
		}

		void seed(std::int32_t seed_value)
		{
			this->seed_with_binary_string(std::bitset<32>(seed_value).to_string());
		}

		void seed(std::uint32_t seed_value)
		{
			this->seed_with_binary_string(std::bitset<32>(seed_value).to_string());
		}

		void seed(std::int64_t seed_value)
		{
			this->seed_with_binary_string(std::bitset<64>(seed_value).to_string());
		}

		void seed(std::uint64_t seed_value)
		{
			this->seed_with_binary_string(std::bitset<64>(seed_value).to_string());
		}

		void seed(const std::string& seed_value)
		{
			this->seed_with_binary_string(seed_value);
		}

		SimulateDoublePendulum()
		{
			this->seed(static_cast<uint64_t>(1));
		}

		explicit SimulateDoublePendulum(auto number)
		{
			this->seed(number);
		}

		~SimulateDoublePendulum()
		{
			this->BackupVelocitys.fill(0.0);
			this->BackupTensions.fill(0.0);
			this->SystemData.fill(0.0);
		}
	};
}

class DirichletApproxOldVersion
{

private:
	long double real_number = 0.0;
	uint64_t limit_count = 0;
	const long double pi = 3.141592653589793;
public:
	// 构造函数，初始化随机数生成器
	DirichletApproxOldVersion() = default;

	using result_type = long double;

	static constexpr result_type min()
	{
		return -2.0;
	}

	static constexpr result_type max()
	{
		return 2.0;
	};

	void reset(long double real_number, uint64_t limit_count)
	{
		this->real_number = real_number;
		this->limit_count = limit_count;
	}

	// 计算Dirichlet函数
	long double operator()()
	{
		if (real_number == 0.0)
		{
			return real_number;
		}

		//y = D(x) =limit[k -> ∞]limit[j -> ∞] cos(k! πx)^{2j}
		for (uint64_t k = 1; k <= limit_count; ++k)
		{
			std::complex<long double> z(0, k * std::tgamma(k + 1) * pi * real_number);  // z(real: 0,imag: j!πx)
			std::complex<long double> e_to_z = std::exp(z);  // e^z(real: 0,imag: j!πx)
			long double result = std::pow(e_to_z.real(), 2 * limit_count);

			if (std::isnan(result) || std::isinf(result))
			{
				// 处理数值不稳定的情况
				return 0.0;
			}
			if (std::isinf(result))
			{
				return 1.0;
			}
			if (std::abs(result) > std::numeric_limits<long double>::epsilon())
			{
				return result;
			}
		}
		return 0.0; // 默认返回值
	}
};

/**
 * @brief  DirichletApprox —— 近似 Dirichlet 函数 D(x) 的可配置类
 *
 * 数学公式（Math）:
 *    D_n(x) = ∏_{k=1}^{n} cos^{2n}(k!·π·x)
 * 重写为：
 *    1. frac_0 = frac(|x|)
 *    2. frac_{k+1} = frac_k * (k+1)  (mod 1)
 *    3. ln D_n(x) = 2n · Σ ln |cos(π·frac_k)|
 *
 * 工程实现（Engineering）:
 *    - 仅用实数运算；无 Gamma / pow 溢出
 *    - 在 log-space 累加避免下溢
 *    - 极小 cos → 直接返回 0，加速收敛
 *
 * 典型用法（Usage）:
 *    DirichletApprox d(32);   // depth = 32
 *    d.reset(0.123456L);      // 设置 x
 *    long double y = d();     // 计算 D_32(x)
 */
class DirichletApprox
{
public:
	using result_type = long double;

	/** @param depth 迭代深度，越大越接近理论 D(x)（>=32 已足够雪崩） */
	explicit DirichletApprox(std::size_t depth = 32) noexcept
		: depth_{ depth } {
	}

	/** 重新设置输入实数 x（Set the target real number x） */
	void reset(long double x) noexcept
	{
		x_ = x;
		// 小数部分；取绝对值避免负号影响 (fractional part of |x|)
		frac_ = std::fabs(x) - std::floor(std::fabs(x));
	}

	static constexpr result_type min()
	{
		return -2.0;
	}

	static constexpr result_type max()
	{
		return 2.0;
	};

	/** 计算近似值（Compute D_depth(x)） */
	[[nodiscard]]
	result_type operator()() const noexcept
	{
		if (x_ == 0.0L)           return 0.0L;  // x = 0, treat as irrational branch
		if (frac_ == 0.0L)        return 1.0L;  // x 为有理数 => D = 1

		constexpr long double pi = 3.141592653589793238462643383279502884L;
		constexpr long double epsilon_threshold = 1e-18L;     // cos 极小阈值 (epsilon_threshold)

		long double frac = frac_;
		long double sum_log_cos = 0.0L;             // ∑ ln |cos|

		for (std::size_t k = 1; k <= depth_; ++k)
		{
			long double c = std::cos(pi * frac);          // cos(π·frac_k)
			if (std::fabs(c) < epsilon_threshold)
				c = epsilon_threshold;

			sum_log_cos += std::log(std::fabs(c));            // ln |cos|

			// frac_{k+1} = (frac_k * (k+1)) mod 1
			frac = std::fmod(frac * static_cast<long double>(k + 1), 1.0L);
		}

		// ln D_n(x) = 2n · sum_log_cos
		long double log_D = 2.0L * static_cast<long double>(depth_) * sum_log_cos;

		// ----- after computing `log_D` -----
		// step 1  (always positive)
		long double v = -log_D;

		// step 2 : fraction( v * phi )
		constexpr long double phi = 0.618033988749894848L;
		// in [0,1)
		long double w = std::fmod(v * phi, 1.0L);

		// step 3 : Weyl double-scramble
		constexpr long double P = 982451653.0L;     // large prime
		long double u = std::fmod(w + std::fmod(w * w * P, 1.0L), 1.0L);

		// final ∈ (0,1)
		return u;
	}

private:
	std::size_t depth_;   // 迭代深度 (iteration depth)
	long double x_{ 0.0L }; // 原始输入 (original x)
	long double frac_{ 0.0L }; // frac(|x|) ∈ [0,1)
};

uint64_t pathosis_hash(uint64_t seed, uint64_t limit_count)
{
	// Seed the random number generator
	std::mt19937_64 prng(1);
	prng.seed(seed);
	// Generate real numbers between lowest and highest double
	ChaoticTheory::SimulateDoublePendulum SDP(static_cast<uint64_t>(prng()));

	//Here, to facilitate the use of c++'s pseudo-random number class, the Dirichlet class function is also encapsulated in the form of a class.
	//这里为了方便使用c++的伪随机数类，所以把Dirichlet函数也封装成了类的形式。
	DirichletApproxOldVersion dirichlet;

	uint64_t result = 0;

	for (int i = 0; i < 128; ++i)
	{
#if 1
		long double real_number = (std::sin(SDP()) + 1.0 / 2.0) * (1024.0e64 - 1024.0e-64) + 1024.0e-64;
#else
		long double real_number = SDP() * (1024.0e64 - 1024.0e-64) + 1024.0e-64;
#endif
		//std::cout << "real_number: "<< real_number << std::endl;

		dirichlet.reset(real_number, limit_count);
		long double dirichlet_value = dirichlet();

		//Uniform01
		long double uniform01_value = 0.0;
		if (dirichlet_value >= 0.0 && dirichlet_value <= 1.0)
		{
			if ((dirichlet_value - 0.0 > std::numeric_limits<long double>::epsilon()) || (dirichlet_value - 1.0 > std::numeric_limits<long double>::epsilon()))
			{
				uniform01_value = dirichlet_value;
			}
		}
		else
		{
			if (dirichlet_value < 0.0)
			{
				uniform01_value = fmod(-dirichlet_value * 4294967296, 1.0);
			}
			else if (dirichlet_value > 1.0)
			{
				uniform01_value = fmod(dirichlet_value * 4294967296, 1.0);
			}
		}

		//if(uniform01_value < 0.0 || uniform01_value > 1.0)
			//std::cerr << "uniform01_value range is error! " << std::endl;
		//std::cout << "uniform01_value: " << uniform01_value << std::endl;

		result |= (uint64_t)(uniform01_value >= 0.5) << i;
	}

	return result;
}

/*────────────────────────────────────────────────────────
 *  Pathosis-Hash  (64-bit 版本)
 *  修复点：
 *    ① result 只用 64bit → i < 64, 无 UB
 *    ② SDP() 归一化到 [0,1)，避免极端放缩
 *    ③ uniform01 简化：直接 |fmod(value,1)|
 *────────────────────────────────────────────────────────*/
inline uint64_t pathosis_hash_new_version(uint64_t seed, std::size_t depth = 32)
{
	/* 1. 统一熵入口 (single entropy source) */
	std::mt19937_64 prng(seed);                   // CSPRNG 可换 random_device
	ChaoticTheory::SimulateDoublePendulum sdp(prng());

	/* 2. Dirichlet 逼近器，depth 对应 limit_count */
	DirichletApprox dirichlet_approx(depth);

	uint64_t result = 0ULL;

	for (std::size_t i = 0; i < 64; ++i)          // *** 不再越界移位 ***
	{
		// 2.1 读摆值并展开成“指数型实数”
		long double raw = std::fabsl(sdp());           // 任意正数
		int exp2 = 0;
		long double mant = std::frexpl(raw, &exp2);     // mant ∈ [0.5,1)
		// 把 2^exp2 映射成 3^exp2，保持数值在 double 可表示范围
		long double x = mant * std::powl(3.0L, exp2);

		// 2.2 放给 Dirichlet
		dirichlet_approx.reset(x);
		long double dirichlet_value = dirichlet_approx();         // 0 – 1 随机分布

		// 2.3 直接用 dirichlet_value，无须再 fmod
		result |= (static_cast<uint64_t>(dirichlet_value >= 0.5L) << i);
	}
	return result;
}

/*────────────────────────────────────────────────────────
 *  Pathosis-Hash  (128-bit 版本，可选)
 *────────────────────────────────────────────────────────*/
inline std::array<uint64_t, 2> pathosis_hash128_new_version(uint64_t seed, std::size_t depth = 32)
{
#if defined(__SIZEOF_INT128__)          // GCC/Clang 支持 __int128
	unsigned __int128 result = 0;

	/* 1. 统一熵入口 (single entropy source) */
	std::mt19937_64 prng(seed);
	ChaoticTheory::SimulateDoublePendulum sdp(prng());

	/* 2. Dirichlet 逼近器，depth 对应 limit_count */
	DirichletApprox dirichlet_approx(depth);

	for (std::size_t i = 0; i < 128; ++i)
	{
		// 2.1 读摆值并展开成“指数型实数”
		long double raw = std::fabsl(sdp());           // 任意正数
		int exp2 = 0;
		long double mant = std::frexpl(raw, &exp2);     // mant ∈ [0.5,1)
		// 把 2^exp2 映射成 3^exp2，保持数值在 double 可表示范围
		long double x = mant * std::powl(3.0L, exp2);

		// 2.2 放给 Dirichlet
		dirichlet_approx.reset(x);
		long double dirichlet_value = dirichlet_approx();         // 0 – 1 随机分布
		result |= (static_cast<unsigned __int128>(dirichlet_value >= 0.5L) << i);
	}
	return { static_cast<uint64_t>(result), static_cast<uint64_t>(result >> 64) };
#else
	/* MSVC 或不支持 __int128 时，退化为两次 64-bit 调用 */
	return { pathosis_hash(seed, depth), pathosis_hash(seed ^ 0xA5A5A5A5A5A5A5A5ULL, depth) };
#endif
}

inline void print_pathosis_hash()
{
	//std::cout << std::setiosflags(std::ios::fixed);
	//std::cout << std::setprecision(24);
	std::mt19937_64 prng(2);
	uint64_t rounds = 256, sub_rounds = 102400;

	uint64_t hash_number_a = 0;
	uint64_t hash_number_b = 0;
	std::string binary_string_a = "";
	std::string binary_string_b = "";
	std::string binary_string_a_xor_b = "";

	for (uint64_t round = 0; round < rounds; round++)
	{
		//std::cout << pathosis_hash(prng(), sub_rounds) << std::endl;
		hash_number_a = pathosis_hash_new_version(prng(), sub_rounds);
		hash_number_b = pathosis_hash_new_version(prng(), sub_rounds);

		std::cout << "pathosis_hash a: " << hash_number_a << std::endl;
		std::cout << "pathosis_hash b: " << hash_number_b << std::endl;

		binary_string_a = std::bitset<64>(hash_number_a).to_string();
		std::cout << "binary_string a: " << binary_string_a << std::endl;
		binary_string_b = std::bitset<64>(hash_number_b).to_string();
		std::cout << "binary_string b: " << binary_string_b << std::endl;
		binary_string_a_xor_b = (std::bitset<64>(hash_number_a) ^ std::bitset<64>(hash_number_b)).to_string();
		std::cout << "binary_string a xor b: " << binary_string_a_xor_b << std::endl;
		//sub_rounds++;
	}

}

inline void hash_chain_with_pathosis_hash()
{
	//std::cout << std::setiosflags(std::ios::fixed);
	//std::cout << std::setprecision(24);
	std::mt19937_64 prng(2);
	uint64_t rounds = 256, sub_rounds = 102400;

	uint64_t hash_number_a = 0;
	uint64_t hash_number_b = 0;
	std::string binary_string_a = "";
	std::string binary_string_b = "";
	std::string binary_string_a_xor_b = "";

	for (uint64_t round = 0; round < rounds; round++)
	{
		if (round == 0)
		{
			hash_number_a = pathosis_hash_new_version(prng(), sub_rounds);
			hash_number_b = pathosis_hash_new_version(prng(), sub_rounds);
		}
		else
		{
			hash_number_a = pathosis_hash_new_version(hash_number_b, sub_rounds);
			hash_number_b = pathosis_hash_new_version(hash_number_a, sub_rounds);
		}

		std::cout << "pathosis_hash a: " << hash_number_a << std::endl;
		std::cout << "pathosis_hash b: " << hash_number_b << std::endl;

		binary_string_a = std::bitset<64>(hash_number_a).to_string();
		std::cout << "binary_string a: " << binary_string_a << std::endl;
		binary_string_b = std::bitset<64>(hash_number_b).to_string();
		std::cout << "binary_string b: " << binary_string_b << std::endl;
		binary_string_a_xor_b = (std::bitset<64>(hash_number_a) ^ std::bitset<64>(hash_number_b)).to_string();
		std::cout << "binary_string a xor b: " << binary_string_a_xor_b << std::endl;
	}
}

inline void arx_with_pathosis_hash()
{
	//std::cout << std::setiosflags(std::ios::fixed);
	//std::cout << std::setprecision(24);
	std::mt19937_64 prng(2);
	uint64_t rounds = 256, sub_rounds = 102400;

	uint64_t hash_number_a = pathosis_hash_new_version(prng(), sub_rounds);
	uint64_t hash_number_b = pathosis_hash_new_version(prng(), sub_rounds);
	uint64_t hash_number_c = pathosis_hash_new_version(prng(), sub_rounds);
	uint64_t hash_number_d = pathosis_hash_new_version(prng(), sub_rounds);

	std::cout << "pathosis_hash a: " << hash_number_a << std::endl;
	std::cout << "pathosis_hash b: " << hash_number_b << std::endl;
	std::cout << "pathosis_hash c: " << hash_number_c << std::endl;
	std::cout << "pathosis_hash d: " << hash_number_d << std::endl;

	std::string binary_string_a = "";
	std::string binary_string_b = "";
	std::string binary_string_a_xor_b = "";

	for (uint64_t round = 0; round < rounds; round++)
	{
		//Add-(Bits)Rotate-Xor Example:
		uint64_t OperatorA = hash_number_a + hash_number_b;
		uint64_t OperatorX = hash_number_c ^ hash_number_d;
		uint64_t OperatorR = OperatorA ^ std::rotl(OperatorA, 47) ^ OperatorX;

		hash_number_a = OperatorR;
		hash_number_b = OperatorX;
		hash_number_c = OperatorA;
		hash_number_d = std::rotl((OperatorR + OperatorX), 64 - 47) ^ OperatorA;

		std::cout << "arx_with_pathosis_hash a: " << hash_number_a << std::endl;
		std::cout << "arx_with_pathosis_hash b: " << hash_number_b << std::endl;
		std::cout << "arx_with_pathosis_hash c: " << hash_number_c << std::endl;
		std::cout << "arx_with_pathosis_hash d: " << hash_number_d << std::endl;

		binary_string_a = std::bitset<64>(hash_number_a).to_string();
		std::cout << "binary_string a: " << binary_string_a << std::endl;
		binary_string_b = std::bitset<64>(hash_number_b).to_string();
		std::cout << "binary_string b: " << binary_string_b << std::endl;
		binary_string_a = std::bitset<64>(hash_number_c).to_string();
		std::cout << "binary_string c: " << binary_string_a << std::endl;
		binary_string_b = std::bitset<64>(hash_number_d).to_string();
		std::cout << "binary_string d: " << binary_string_b << std::endl;

		hash_number_a = pathosis_hash(hash_number_a, sub_rounds);
		hash_number_b = pathosis_hash(hash_number_b, sub_rounds);
		hash_number_c = pathosis_hash(hash_number_c, sub_rounds);
		hash_number_d = pathosis_hash(hash_number_d, sub_rounds);
	}
}

int main()
{
	print_pathosis_hash();
	//hash_chain_with_pathosis_hash();
	//arx_with_pathosis_hash();

	return 0;
}
