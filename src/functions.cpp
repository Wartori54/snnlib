#include <functions.h>
#include <random>
#include <chrono>
#include <stdexcept>
#include <cmath>

#define vec_size std::vector::size_

void invalid_dfunc(std::vector<neuron_state>& pond_sums, std::vector<double>& outs) {
    throw std::runtime_error("invalid_dfunc was called!!");
    pond_sums[0].result = outs[0]; // unreachable statement, only to avoid compiler warnings
}

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

jacob_act_func ActivationFunctions::softmax = {
    ActivationFunctions::fsoftmax,
    ActivationFunctions::djsoftmax,
};

void ActivationFunctions::fsoftmax(std::vector<neuron_state>& outs) {
    static std::vector<double> pows(outs.size()); // temporal array, static to prevent deletion because it can be reused
    if (outs.size() > pows.size()) { // will always be equal or bigger than outs, for speed purposes 
        pows.resize(outs.size());
    }
    double mx = 0; // biggest element
    for (size_t i = 0; i < outs.size(); i++) {
        if (outs[i].pond_sum > mx) mx = outs[i].pond_sum;
    }
    double sum = 0; // denominator calculation
    double cval;
    for (size_t i = 0; i < outs.size(); i++) {
        cval = std::pow(M_E, outs[i].pond_sum-mx);
        sum += cval;
        pows[i] = cval;
    }
    for (size_t i = 0; i < outs.size(); i++)
        outs[i].result = pows[i]/sum;
}

void ActivationFunctions::djsoftmax(std::vector<neuron_state>& pond_sums, std::vector<std::vector<double>>& outs) {
    for (size_t i = 0; i < pond_sums.size(); i++) {
        for (size_t j = 0; j < pond_sums.size(); j++) {
            // here .result is used because the derivative is softmax(x)[i*(1-softmax(x)[i]). 
            //softmax(x) is already calculated and its stored in pond_sums[i].result
            if (i == j) { 
                outs[i][j] = pond_sums[i].result*(1-pond_sums[j].result);
            } else {
                outs[i][j] = pond_sums[i].result*(0-pond_sums[j].result);
            }
        }
    }
}

InitialitzationFunctions::StaticInitFunc::StaticInitFunc(double weight_val, double bias_val) {
    w_val = weight_val;
    b_val = bias_val;
};

void InitialitzationFunctions::StaticInitFunc::do_init(weight_vec& weights, bias_vec& biases) {
    for (size_t i = 0; i < weights.size(); i++) {
        std::normal_distribution<double> distr(0, sqrt(2.0/weights[i].size()));
        for (size_t j = 0; j < weights[i].size(); j++) {
            for (size_t k = 0; k < weights[i][j].size(); k++) {
                weights[i][j][k] = w_val;
            }
        }
    }
    for (size_t i = 1; i < biases.size(); i++)
        for (size_t j = 0; j < biases[i].size(); j++) {
            biases[i][j] = b_val; // biases at 0
        }
}

InitialitzationFunctions::RandomInitFunc::RandomInitFunc(uint_fast32_t seed) {
    this->eng = std::default_random_engine(rd());
    set_seed(seed);
}

void InitialitzationFunctions::RandomInitFunc::set_seed(uint_fast32_t seed) {
    this->eng.seed(seed);
}

InitialitzationFunctions::HeInit::HeInit(uint_fast32_t seed) : RandomInitFunc(seed) {
}

void InitialitzationFunctions::HeInit::do_init(weight_vec& weights, bias_vec& biases) {
    for (size_t i = 0; i < weights.size(); i++) {
        std::normal_distribution<double> distr(0, sqrt(2.0/weights[i].size()));
        for (size_t j = 0; j < weights[i].size(); j++) {
            for (size_t k = 0; k < weights[i][j].size(); k++) {
                weights[i][j][k] = distr(this->eng);
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

error_func ErrorFunctions::cross_entropy = {
    ErrorFunctions::fcross_entropy,
    ErrorFunctions::dcross_entropy
};

double ErrorFunctions::fcross_entropy(std::vector<neuron_state>& res, std::vector<double>& target) {
    double out = 0;
    if (res.size() != target.size()) {
        throw std::runtime_error("res and target are not same size");
    }
    for (size_t i = 0; i < res.size(); i++) {
        out += target[i] * std::log(res[i].result);
    }
    return -out;
}

void ErrorFunctions::dcross_entropy(std::vector<neuron_state>& res, 
                std::vector<double>& target, 
                std::vector<double>& out) {
    
    for (size_t i = 0; i < res.size(); i++) 
        out[i] = -target[i]/res[i].result;
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