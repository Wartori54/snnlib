#pragma once

#include <vector>
#include <memory>
#include <optional>
#include <functions.h>

namespace NeuralNetwork {
    class FFNeuralNetwork;

    /// @brief The base class that every implementation should extend.
    /// **DO NO INSTANCIATE THIS** -- because you can't
    class GradientDescendOptim {
    protected:
        std::shared_ptr<FFNeuralNetwork> net_owner = nullptr;
        // notation: L -> the cost function
        //           a -> activation function
        //           z -> the neuron result before activation function
        //           w -> a certain weight
        //           b -> a certain bias
        std::vector<double> dLofa;      //              dL/da -> derivative of L in respect to a
        std::vector<double> temp_dLofa; //              copy for gradient_descent
        std::vector<std::vector<double>> daofz; //       da/dz, 2d array for support for jacobian matrix
        // we skip dzofw and dzofb as they would be simple or constant values
        std::vector<std::vector<double>> dLofw; // dL/dw -> (dL/da)*(da/dz)*(dz/dw). 
        // dz/dw always turns out to be the activation of the previous neurons
        std::vector<double> dLofb; //              dL/dw -> (dL/da)*(da/dz)*(dz/db). dz/dw always turns out to be 1
    public:
        ~GradientDescendOptim();
        /// @brief Must be called before using at all.
        /// @param net The network that this object is attached to.
        virtual void set_owner(std::shared_ptr<FFNeuralNetwork> net);
        /// @brief Proceeds to do the next iteration of the algorithm.
        /// @param targets The expected output values for the current case.
        void step(std::vector<double> &targets);
        virtual GradientDescendOptim* make_copy() = 0;
    private:
        void gradient_descent(std::vector<double> &targets);
        virtual void shift_values(int wl) = 0;
    };

    /// @brief Standard Backpropagation.
    class BProp : public GradientDescendOptim {
    private:
        double learning_rate = 0;
        std::unique_ptr<OptimizerFunctions::LearningRateOptimizerBase> lr_func;
    public:
        /// @brief Creates an instance of BProp.
        /// @param learning_rate The learning rate for the backpropagation.
        /// @param lr_func A LearningRateOptimizerBase instance to modify the learning_rate progressively.
        BProp(double learning_rate, OptimizerFunctions::LearningRateOptimizerBase& lr_func);
        ~BProp();
        GradientDescendOptim* make_copy();
    private:
        void shift_values(int wl);
    };

    /// @brief The Adam gradient descend algorithm.
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
        /// @brief Creates an instance of Adam.
        /// @param alpha Parameter alpha. Commonly assigned to 0.002.
        /// @param beta1 Parameter alpha. Commonly assigned to 0.9.
        /// @param beta2 Parameter alpha. Commonly assigned to 0.999.
        /// @param elipson Small shift denominator to prevent division by zero. Commonly assigned to 1e-8.
        Adam(double alpha, double beta1, double beta2, double elipson);
        ~Adam();
        // See superclass doc.
        void set_owner(std::shared_ptr<FFNeuralNetwork> net);
        GradientDescendOptim* make_copy();
    private:
        void shift_values(int wl);
    };

    class NNConfig {
        public:
        std::vector<int> n_layers;
        int n_inputs, n_outputs;
        std::vector<int> h_layers;
        std::vector<act_func*> act_funcs;
        std::shared_ptr<InitialitzationFunctions::InitFunc> v_init;
        NNConfig(int n_inputs,
                 int n_outputs,
                 std::vector<int> h_layers,
                 std::vector<act_func*> act_funcs,
                 std::shared_ptr<InitialitzationFunctions::InitFunc> v_init); // here we can afford a copy of all vectors as they wont be too big
    };

    /**
     * @brief Defines the design of a neural network.
     * 
     */
    class FFNNConfig : public NNConfig {
    public:
        error_func e_func;
        bool verbose = true;
        ///
        /// @brief Construct a new FFNNConfig, used to store neural network design.
        /// 
        /// @param n_inputs number of inputs.
        /// @param n_outputs number of outputs.
        /// @param h_layers array describing the hidden layers, each value is the number of neurons in that layer.
        /// @param act_funcs activation_functions of every layer, skipping input layer, must have size of h_layers.size()+1.
        /// @param v_init the initialtization function for the weights and biases.
        ///
        FFNNConfig(int n_inputs,
                 int n_outputs,
                 std::vector<int> h_layers,
                 std::vector<act_func*> act_funcs,
                 std::shared_ptr<InitialitzationFunctions::InitFunc> v_init,
                 error_func e_func); // here we can afford a copy of all vectors as they wont be too big
    };

    class NeuralNet {
    protected:
    public:
        std::shared_ptr<NNConfig> conf;
        weight_vec weights;
        bias_vec biases;
        weight_vec s_weights; // weight shifts
        bias_vec s_biases; // biases shifts
        activations_vec activs;
    protected:
        NeuralNet(NNConfig& conf);
    };

    /// @brief A feed forward neural network.
    class FFNeuralNetwork : public NeuralNet {
    private:
    public:
        std::shared_ptr<GradientDescendOptim> optimizer;
        std::shared_ptr<FFNNConfig> config;
        
        unsigned long long iteration;
        /// @brief Creates a feed foward neural network.
        /// @param conf The config that defines the network.
        /// @param optim The algorithm to use for bias and weight manipulation.
        static std::shared_ptr<FFNeuralNetwork> create(FFNNConfig& conf, GradientDescendOptim& optim);
        ~FFNeuralNetwork();

        /// @brief Obtains the config of the network.
        /// @return The assigned FFNNConfig instance.
        std::shared_ptr<NeuralNetwork::FFNNConfig> get_config();
        /// @brief Does a forward pass.
        /// @param inputs The inputs to predict on.
        /// @return The results at the output layer.
        std::vector<neuron_state>& predict(std::vector<double>& inputs);
        /// @brief Trains the neural network using the assigned algorithm.
        /// @param inputs The inputs of the dataset to train on.
        /// @param targets The target outputs of the dataset to train on.
        /// @param batch_s The mini batch size. The network will be updated every batch_s dataset samples .
        /// @param batches The total number of interations to the whole dataset.
        /// @return The cost of the last batch
        double fit(std::vector<std::vector<double>>& inputs, std::vector<std::vector<double>>& targets, size_t batch_s, size_t batches);
    protected:
        FFNeuralNetwork(FFNNConfig& conf, GradientDescendOptim& optim);
    };
}
