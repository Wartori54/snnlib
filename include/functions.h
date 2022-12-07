#pragma once

#include <vector>
#include <random>
#include <stdexcept>

struct neuron_state {
    double pond_sum;
    double result;
};

struct act_func { // both functions should read from neuron_state.pond_sum and assings the neuron_state.result
    void (*func)(std::vector<neuron_state>& outs); // activation func, takes all results and gives outputs
    void (*dfunc)(std::vector<neuron_state>& pond_sums, std::vector<double>& outs); // derivateive of the act func
};

struct error_func {
    double (*func)(std::vector<neuron_state>& res, std::vector<double>& target); // res is different for target for convenience
    void (*dfunc)(std::vector<neuron_state>& res, 
                    std::vector<double>& target, 
                    std::vector<double>& out); // out should always have enough size
};


typedef std::vector<std::vector<std::vector<double>>> weight_vec;
typedef std::vector<std::vector<double>> bias_vec;
typedef std::vector<std::vector<neuron_state>> activations_vec;

namespace ActivationFunctions {

    void flinear(std::vector<neuron_state>& outs);
    void dlinear(std::vector<neuron_state>& pond_sums, std::vector<double>& outs);
    
    extern act_func linear;

    void frelu(std::vector<neuron_state>& outs);
    void drelu(std::vector<neuron_state>& pond_sums, std::vector<double>& outs);

    extern act_func relu;

    void fsigmoid(std::vector<neuron_state>& outs);
    void dsigmoid(std::vector<neuron_state>& pond_sums, std::vector<double>& outs);

    extern act_func sigmoid;

} // namespace ActivationFunctions

namespace InitialitzationFunctions {
    extern std::random_device rd;
    extern std::default_random_engine eng;

    extern bool is_setup;

    void setup();
    void setup(long seed);

    void he_init(weight_vec& weights, bias_vec& biases);
    
} // namespace InitialitzationFunctions

namespace ErrorFunctions {
    extern error_func mse;

    double fmse(std::vector<neuron_state>& res, std::vector<double>& target);
    void dmse(std::vector<neuron_state>& res, 
                    std::vector<double>& target, 
                    std::vector<double>& out);
} // namespace ErrorFunctions

namespace OptimizerFunctions {

    class LearningRateOptimizerBase {
    private:
    public:
        LearningRateOptimizerBase();
        ~LearningRateOptimizerBase();
        virtual double step(double lr, unsigned long long itr) {
            lr = itr; itr = lr; // trick compiler to remove unused parameter warnings
            throw std::runtime_error("step: unimplemented");
        };
    };

    class LRDecay : public LearningRateOptimizerBase {
    private:
        double decay;
    public:
        LRDecay(double decay);
        ~LRDecay();
        double step(double lr, unsigned long long itr);
    };

} // namespace OptimizerFunctions
