#pragma once

#include <vector>
#include <memory>
#include <functions.h>

namespace NeuralNetwork {
    class FFNeuralNetwork;

    class GradientDescendOptim {
    protected:
        std::unique_ptr<FFNeuralNetwork> net_owner;
        // notation: L -> the cost function
        //           a -> activation function
        //           z -> the neuron result before activation function
        //           w -> a certain weight
        //           b -> a certain bias
        std::vector<double> dLofa;      //              dL/da -> derivative of L in respect to a
        std::vector<double> temp_dLofa; //              copy for gradient_descent
        std::vector<double> daofz;      //              da/dz
        // we skip dzofw and dzofb as they would be simple or constant values
        std::vector<std::vector<double>> dLofw; // dL/dw -> (dL/da)*(da/dz)*(dz/dw). 
        // dz/dw always turns out to be the activation of the previous neurons
        std::vector<double> dLofb; //              dL/dw -> (dL/da)*(da/dz)*(dz/db). dz/dw always turns out to be 1
    public:
        virtual void set_owner(FFNeuralNetwork *net);
        void step(std::vector<double> &targets);
    private:
        void gradient_descent(std::vector<double> &targets);
        virtual void shift_values(int wl) {
            wl = wl;
            throw std::runtime_error("shift_values: unimplemented");
        };
    };

    class BProp : public GradientDescendOptim {
    private:
        double learning_rate = 0;
        std::unique_ptr<OptimizerFunctions::LearningRateOptimizerBase> lr_func;
    public:
        BProp(double learning_rate, OptimizerFunctions::LearningRateOptimizerBase *lr_func);
        ~BProp();
    private:
        void shift_values(int wl);
    };

    class Adam : public GradientDescendOptim {
    private:
        double alpha;
        double beta1;
        double beta2;
        double pow_beta1;
        double pow_beta2;
        unsigned long long last_itr;
        double elipson;
        weight_vec m_w; // first moment weights
        weight_vec v_w; // second moment weights
        bias_vec m_b; // first moment bias
        bias_vec v_b; // second moment bias
    public:
        Adam(double alpha, double beta1, double beta2, double elipson);
        ~Adam();
        void shift_values(int wl);
        void set_owner(FFNeuralNetwork *net);
    };
    

    /**
     * @brief Defines the design of a neural network
     * 
     */
    class NNConfig {
    public:
        std::vector<int> n_layers;
        int n_inputs, n_outputs;
        std::vector<int> h_layers;
        std::vector<act_func> act_funcs;
        error_func e_func;
        void (*v_init)(weight_vec& weights, bias_vec& biases);
        ///
        /// @brief Construct a new NNConfig, used to store neural network design
        /// 
        /// @param n_inputs number of inputs
        /// @param n_outputs number of outputs
        /// @param h_layers array describing the hidden layers, each value is the number of neurons in that layer
        /// @param act_funcs activation_functions of every layer, skipping input layer, must have size of h_layers.size()+1
        /// @param v_init the initialtization function for the weights and biases
        ///
        NNConfig(int n_inputs,
                 int n_outputs,
                 std::vector<int> h_layers,
                 std::vector<act_func> act_funcs,
                 void (*v_init)(weight_vec&, bias_vec&),
                 error_func e_func); // here we can afford a copy of all vectors as they wont be too big
        ~NNConfig();
    };

    class FFNeuralNetwork {
    private:
    public:
        std::unique_ptr<NNConfig> config;
        std::unique_ptr<GradientDescendOptim> optimizer;
        weight_vec weights;
        bias_vec biases;
        weight_vec s_weights; // weight shifts
        bias_vec s_biases; // biases shifts
        activations_vec activs;
        unsigned long long iteration;
        FFNeuralNetwork(NNConfig *conf, GradientDescendOptim *optim);
        ~FFNeuralNetwork();
        NNConfig *get_config();
        std::vector<neuron_state>& predict(std::vector<double>& inputs);
        void fit(std::vector<std::vector<double>>& inputs, std::vector<std::vector<double>>& targets, size_t batch_s, size_t batches);
    };
}
