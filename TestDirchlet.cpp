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

		std::array<long double, 2> BackupTensions{};
		std::array<long double, 2> BackupVelocitys{};
		std::array<long double, 10> SystemData{};

		static constexpr long double gravity_coefficient = 9.8;
		static constexpr long double hight = 0.002;

		void run_system(bool is_initialize_mode, std::uint64_t time)
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
			this->run_system(true, static_cast<std::uint64_t>(::round(radius * current_binary_key_sequence_size)));
		}

		long double generate()
		{
			//This is generate mode
			this->run_system(false, 1);

			long double temporary_floating_a = 0.0;
			long double temporary_floating_b = 0.0;

			std::int64_t left_number = 0, right_number = 0;

			temporary_floating_a = this->SystemData[0] * ::sin(this->SystemData[4]) + this->SystemData[1] * ::sin(this->SystemData[5]);
			temporary_floating_b = -(this->SystemData[0]) * ::sin(this->SystemData[4]) - this->SystemData[1] * ::sin(this->SystemData[5]);

			temporary_floating_a = fmod(temporary_floating_a * 1000.0, 1.0) * 4294967296;
			temporary_floating_b = fmod(temporary_floating_b * 1000.0, 1.0) * 4294967296;

			if (std::isnan(temporary_floating_a) || std::isnan(temporary_floating_b))
			{
				return 0.0;
			}

			if (std::isinf(temporary_floating_a) || std::isinf(temporary_floating_b))
			{
				return 0.0;
			}

			//std::cout << "temporary_floating_a: " << temporary_floating_a << std::endl;
			//std::cout << "temporary_floating_b: " << temporary_floating_b << std::endl;

			//This is generate mode
			this->run_system(false, 1);
			if (temporary_floating_a * 2.0 + temporary_floating_b >= 0.0)
			{
				return temporary_floating_a;
			}
			else
			{
				return temporary_floating_b;
			}
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

class Dirichlet
{

private:
	long double real_number = 0.0;
	uint64_t limit_count = 0;
	const long double pi = 3.141592653589793;
public:
	// 构造函数，初始化随机数生成器
	Dirichlet() = default;

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

uint64_t pathosis_hash(uint64_t seed, uint64_t limit_count)
{
	// Seed the random number generator
	std::mt19937_64 prng(1);
	prng.seed(seed);
	// Generate real numbers between lowest and highest double
	ChaoticTheory::SimulateDoublePendulum SDP(static_cast<uint64_t>(prng()));

	//Here, to facilitate the use of c++'s pseudo-random number class, the Dirichlet class function is also encapsulated in the form of a class.
	//这里为了方便使用c++的伪随机数类，所以把Dirichlet函数也封装成了类的形式。
	Dirichlet dirichlet;

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
		hash_number_a = pathosis_hash(prng(), sub_rounds);
		hash_number_b = pathosis_hash(prng(), sub_rounds);

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
			hash_number_a = pathosis_hash(prng(), sub_rounds);
			hash_number_b = pathosis_hash(prng(), sub_rounds);
		}
		else
		{
			hash_number_a = pathosis_hash(hash_number_b, sub_rounds);
			hash_number_b = pathosis_hash(hash_number_a, sub_rounds);
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

	uint64_t hash_number_a = pathosis_hash(prng(), sub_rounds);
	uint64_t hash_number_b = pathosis_hash(prng(), sub_rounds);
	uint64_t hash_number_c = pathosis_hash(prng(), sub_rounds);
	uint64_t hash_number_d = pathosis_hash(prng(), sub_rounds);

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
	arx_with_pathosis_hash();

	return 0;
}