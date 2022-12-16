#pragma once

#include <vector>
#include <random>
#include <stdexcept>

/// @brief internal use.
struct neuron_state {
    double pond_sum;
    double result;
};

/// @brief Defines an activation function for a layer.
/// @param func The function in question, reads from @c outs.pond_sum and writes to @c outs.result.
/// @param dfunc The derivative of the function, reads from @c pond_sums.pond_sum and writes to @c outs.
struct act_func { // both functions should read from neuron_state.pond_sum and assings the neuron_state.result.
    void (*func)(std::vector<neuron_state>& outs); // activation func, takes all results and gives outputs
    void (*dfunc)(std::vector<neuron_state>& pond_sums, std::vector<double>& outs); // derivative of the act func
    act_func(void (*f)(std::vector<neuron_state>&), 
             void (*df)(std::vector<neuron_state>& , std::vector<double>& ))
             : func(f), dfunc(df) {};
    virtual ~act_func() {};
};

void invalid_dfunc(std::vector<neuron_state>& pond_sums, std::vector<double>& outs);

struct jacob_act_func : public act_func {
    void (*djfunc)(std::vector<neuron_state>& pond_sums, std::vector<std::vector<double>>& outs);
    jacob_act_func(void (*f)(std::vector<neuron_state>&), 
                   void (*djf)(std::vector<neuron_state>&,  std::vector<std::vector<double>>& )) 
                   : act_func(f, invalid_dfunc), djfunc(djf) {};
};

/// @brief An error calculation function, used to determitante the accuracy of the network on a given case, also used in gradient descent.
/// @param func The function in question, reads from @c res.result and @c target and return a double.
/// @param dfunc The derivative of the function, reads from @c res.result and @c target and writes to @c out.
struct error_func {
    double (*func)(std::vector<neuron_state>& res, std::vector<double>& target); // res is different for target for convenience
    void (*dfunc)(std::vector<neuron_state>& res, 
                    std::vector<double>& target, 
                    std::vector<double>& out); // out should always have enough size
};


typedef std::vector<std::vector<std::vector<double>>> weight_vec;
typedef std::vector<std::vector<double>> bias_vec;
typedef std::vector<std::vector<neuron_state>> activations_vec;

/// @brief The collection of available activation functions.
namespace ActivationFunctions {

    // y = x
    void flinear(std::vector<neuron_state>& outs);
    // y = 1;
    void dlinear(std::vector<neuron_state>& pond_sums, std::vector<double>& outs);
    
    /// @brief Linear activation function.
    /// y = x
    extern act_func linear;

    // y = max(0, x)
    void frelu(std::vector<neuron_state>& outs);
    // y = x > 0 ? 1 : 0
    void drelu(std::vector<neuron_state>& pond_sums, std::vector<double>& outs);

    /// @brief RELU activation function.
    /// y = max(0, x)
    extern act_func relu;

    // y = 1/(1+e^-x)
    void fsigmoid(std::vector<neuron_state>& outs);
    // y = sigmoid(x)*(1-sigmoid(x))
    void dsigmoid(std::vector<neuron_state>& pond_sums, std::vector<double>& outs);

    /// @brief Sigmoid activation function.
    extern act_func sigmoid;

    void fsoftmax(std::vector<neuron_state>& outs);
    void djsoftmax(std::vector<neuron_state>& pond_sums, std::vector<std::vector<double>>& outs);

    extern jacob_act_func softmax;

} // namespace ActivationFunctions

/// @brief The collection of available weight and bias initialitzation algorithms.
namespace InitialitzationFunctions {
    class InitFunc {
        public:
        virtual void do_init(weight_vec& weights, bias_vec& biases) = 0;
    };

    class StaticInitFunc : public InitFunc {
        public:
        double w_val, b_val;
        StaticInitFunc(double weight_val, double bias_val);
        void do_init(weight_vec& weights, bias_vec& biases);
    };
    
    class RandomInitFunc : public InitFunc {
        public:
        std::random_device rd;
        std::default_random_engine eng;
        RandomInitFunc(uint_fast32_t seed);
        void set_seed(uint_fast32_t seed);
    };

    /// @brief The He weight and bias random initalitzation algorithm. Suited for networks were RELU is used.
    /// w_ij = sqrt(2.0/(number on neurons in this layer))
    /// b_i = 0
    class HeInit : public RandomInitFunc {
        public:
        HeInit(uint_fast32_t seed);
        void do_init(weight_vec& weights, bias_vec& biases);
    };
    
} // namespace InitialitzationFunctions

/// @brief The collection of available functions to calculate the accuracy of a network.
namespace ErrorFunctions {
    /// @brief Mean Squared Error function.
    extern error_func mse;

    // err = (y - y')**2
    double fmse(std::vector<neuron_state>& res, std::vector<double>& target);
    // der = -2*(y - y')
    void dmse(std::vector<neuron_state>& res, 
                    std::vector<double>& target, 
                    std::vector<double>& out);

    extern error_func cross_entropy;

    double fcross_entropy(std::vector<neuron_state>& res, std::vector<double>& target);
    void dcross_entropy(std::vector<neuron_state>& res, 
                    std::vector<double>& target, 
                    std::vector<double>& out);
} // namespace ErrorFunctions

/// @brief The collection of available functions to alter the learning rate used in backpropagation and similars.
namespace OptimizerFunctions {

    /// @brief The base class that every implementation should extend.
    /// **DO NO INSTANCIATE THIS**
    class LearningRateOptimizerBase {
    private:
    public:
        LearningRateOptimizerBase();
        ~LearningRateOptimizerBase();
        virtual double step(double lr, unsigned long long itr) = 0;
    };

    /// @brief A basic time-based decay function.
    /// lr_n = lr_n-2*(1+decay*iteration)
    class LRDecay : public LearningRateOptimizerBase {
    private:
        double decay;
    public:
        LRDecay(double decay);
        ~LRDecay();
        double step(double lr, unsigned long long itr);
    };

} // namespace OptimizerFunctions
