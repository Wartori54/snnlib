#include <functions.h>
#include <random>
#include <chrono>
#include <stdexcept>
#include <cmath>

#define vec_size std::vector::size_

/**
  * Linear activation function
  * y = x
  */
act_func ActivationFunctions::linear = {
    ActivationFunctions::flinear,
    ActivationFunctions::dlinear,
};

void ActivationFunctions::flinear(std::vector<neuron_state>& outs) {
    for (size_t i = 0; i < outs.size(); i++) {
        outs[i].result = outs[i].pond_sum;
    }
}

void ActivationFunctions::dlinear(std::vector<neuron_state>& pond_sums, std::vector<double>& outs) {
    for (size_t i = 0; i < pond_sums.size(); i++) 
        outs[i] = 1;
}

/**
  * Relu activation function
  * y = max(0, x)
  */
act_func ActivationFunctions::relu = {
    ActivationFunctions::frelu,
    ActivationFunctions::drelu,
};


void ActivationFunctions::frelu(std::vector<neuron_state>& outs) {
    for (size_t i = 0; i < outs.size(); i++) {
        outs[i].result = outs[i].pond_sum > 0 ? outs[i].pond_sum : 0;
    }
}

void ActivationFunctions::drelu(std::vector<neuron_state>& pond_sums, std::vector<double>& outs) {
    for (size_t i = 0; i < pond_sums.size(); i++) 
        outs[i] = pond_sums[i].pond_sum > 0 ? 1 : 0;
}

act_func ActivationFunctions::sigmoid = {
    ActivationFunctions::fsigmoid,
    ActivationFunctions::dsigmoid,
};

/**
  * Sigmoid activation function
  * y = 1/(1+e^-x)
  */
void ActivationFunctions::fsigmoid(std::vector<neuron_state>& outs) {
    for (size_t i = 0; i < outs.size(); i++) {
        outs[i].result = 1/(1+std::pow(M_E, -outs[i].pond_sum));
    }
}

void ActivationFunctions::dsigmoid(std::vector<neuron_state>& pond_sums, std::vector<double>& outs) {
    double res = 0;
    for (size_t i = 0; i < pond_sums.size(); i++) {
        res = 1/(1+std::pow(M_E, -pond_sums[i].pond_sum));
        outs[i] = res*(1-res);
    }
}

std::random_device InitialitzationFunctions::rd;
std::default_random_engine InitialitzationFunctions::eng(rd());

bool InitialitzationFunctions::is_setup = false;

void InitialitzationFunctions::setup(long seed) {
    InitialitzationFunctions::eng.seed(seed);
    InitialitzationFunctions::is_setup = true;
}

void InitialitzationFunctions::setup() {
    InitialitzationFunctions::setup(std::chrono::system_clock::now().time_since_epoch().count());
}

void check_setup() {
    if (!InitialitzationFunctions::is_setup) {
        throw std::runtime_error("InitialitzationFunctions::setup() was not called!");
    }
}

void InitialitzationFunctions::he_init(weight_vec& weights, bias_vec& biases) {
    check_setup();
    for (size_t i = 0; i < weights.size(); i++) {
        std::normal_distribution<double> distr(0, sqrt(2.0/weights[i].size()));
        for (size_t j = 0; j < weights[i].size(); j++) {
            for (size_t k = 0; k < weights[i][j].size(); k++) {
                weights[i][j][k] = distr(InitialitzationFunctions::eng);
            }
        }
    }
    for (size_t i = 1; i < biases.size(); i++)
        for (size_t j = 0; j < biases[i].size(); j++) {
            biases[i][j] = 0; // biases at 0
        }
}

error_func ErrorFunctions::mse = {
    ErrorFunctions::fmse,
    ErrorFunctions::dmse
};

double ErrorFunctions::fmse(std::vector<neuron_state>& res, std::vector<double>& target) {
    double out = 0;
    if (res.size() != target.size()) {
        throw std::runtime_error("res and target are not same size");
    }
    for (size_t i = 0; i < res.size(); i++) {
        out += pow((target[i]-res[i].result), 2);
    }
    out /= res.size();
    return out;
}

void ErrorFunctions::dmse(std::vector<neuron_state>& res, 
          std::vector<double>& target, 
          std::vector<double>& out) {
    for (size_t i = 0; i < res.size(); i++) 
        out[i] = -2*(target[i]-res[i].result);
}

OptimizerFunctions::LearningRateOptimizerBase::LearningRateOptimizerBase() {}

OptimizerFunctions::LearningRateOptimizerBase::~LearningRateOptimizerBase() {}

OptimizerFunctions::LRDecay::LRDecay(double decay) {
    this->decay = decay;
}

OptimizerFunctions::LRDecay::~LRDecay() {
}

double OptimizerFunctions::LRDecay::step(double lr, unsigned long long itr) {
    return lr/(1+this->decay*itr);
}