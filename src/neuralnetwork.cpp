#include <algorithm>
#include <iterator>
#include <vector>
#include <string>
#include <iostream>
#include <chrono>
#include <iomanip>

#include <neuralnetwork.h>
#include <functions.h>


NeuralNetwork::NNConfig::NNConfig(int n_inputs, int n_outputs,
                                  std::vector<int> h_layers,
                                  std::vector<act_func*> act_funcs,
                                  std::shared_ptr<InitialitzationFunctions::InitFunc> v_init) {
    if (h_layers.size()+1 != act_funcs.size()) 
        throw std::length_error("h_layers and act_funcs size mismatch, sizes: " + std::to_string(h_layers.size()) + ", " + std::to_string(act_funcs.size()));
    this->n_inputs = n_inputs;
    this->n_outputs = n_outputs;
    this->h_layers = h_layers;
    this->act_funcs = act_funcs;
    this->v_init = v_init;
    this->n_layers = std::vector(h_layers.size()+2, 0);
    this->n_layers[0] = n_inputs;
    this->n_layers[n_layers.size()-1] = n_outputs;
    for (size_t i = 0; i < h_layers.size(); i++) {
        this->n_layers[i+1] = h_layers[i];
    }
}

NeuralNetwork::FFNNConfig::FFNNConfig(int n_inputs, 
                 int n_outputs, 
                 std::vector<int> h_layers, 
                 std::vector<act_func*> act_funcs, 
                 std::shared_ptr<InitialitzationFunctions::InitFunc> v_init,
                 error_func e_func) : NNConfig(n_inputs, n_outputs, h_layers, act_funcs, v_init) {
    this->e_func = e_func;   
}

NeuralNetwork::NeuralNet::NeuralNet(NNConfig& conf) {
    this->conf = std::shared_ptr<NNConfig>(new NNConfig(conf));
    // setup arrays
    weights.resize(conf.n_layers.size()-1);
    biases.resize(conf.n_layers.size()); // biases[0] will be unused
    activs.resize(conf.n_layers.size());
    for (size_t i = 0; i < conf.n_layers.size()-1; i++) {
        weights[i].resize(conf.n_layers[i]);
        for (int j = 0; j < conf.n_layers[i]; j++) {
            weights[i][j] = std::vector<double>(conf.n_layers[i+1], 0);
        }

    }
    activs[0] = std::vector<neuron_state>(conf.n_inputs, {0, 0});
    for (size_t i = 1; i < conf.n_layers.size(); i++) {
        biases[i] = std::vector<double>(conf.n_layers[i], 0);
        activs[i] = std::vector<neuron_state>(conf.n_layers[i], {0, 0});
    }

    // init network
    conf.v_init->do_init(weights, biases);
    // create temp weights and biases
    s_biases = biases;
    s_weights = weights;
}

std::shared_ptr<NeuralNetwork::FFNeuralNetwork> NeuralNetwork::FFNeuralNetwork::create(FFNNConfig& conf, GradientDescendOptim& optim) {
    auto temp = std::shared_ptr<FFNeuralNetwork>(new FFNeuralNetwork(conf, optim));
    temp->optimizer->set_owner(temp);
    return temp;
}

NeuralNetwork::FFNeuralNetwork::FFNeuralNetwork(FFNNConfig& conf, GradientDescendOptim& optim) : NeuralNetwork::NeuralNet(conf) {
    this->optimizer = std::shared_ptr<GradientDescendOptim>(optim.make_copy());
    this->iteration = 0;
    this->config = std::shared_ptr<FFNNConfig>(new FFNNConfig(conf));
}

NeuralNetwork::FFNeuralNetwork::~FFNeuralNetwork() {
}

std::shared_ptr<NeuralNetwork::FFNNConfig> NeuralNetwork::FFNeuralNetwork::get_config() {
    return this->config;
}

std::vector<neuron_state>& NeuralNetwork::FFNeuralNetwork::predict(std::vector<double>& inputs) {
    for (size_t i = 0; i < inputs.size(); i++)
        activs[0][i] = {inputs[i], inputs[i]}; // set up input
    // iterate over all layers
    for (size_t l = 1; l < config->n_layers.size(); l++) {
        for (int n_i = 0; n_i < config->n_layers[l]; n_i++) { // go through each neuron
            activs[l][n_i].pond_sum = 0;
            for (int w = 0; w < config->n_layers[l-1]; w++) { // go through each weight
                activs[l][n_i].pond_sum += weights[l-1][w][n_i] * activs[l-1][w].result;
                // important notes:
                // weights[starting layer][start][end]
            }
            activs[l][n_i].pond_sum += biases[l][n_i];
        }
        config->act_funcs[l-1]->func(activs[l]);
    }
    
    return activs[config->n_layers.size()-1];
}

double NeuralNetwork::FFNeuralNetwork::fit(std::vector<std::vector<double>>& inputs, std::vector<std::vector<double>>& targets, size_t batch_s, size_t batches) {
    if (inputs.size() != targets.size()) {
        throw std::runtime_error("fit: inputs.size() != targets.size()");
    }
    size_t total_samples = inputs.size();
    size_t total_splits = (total_samples/batch_s)+1;
    size_t lbatch_size = total_samples-batch_s*(total_splits-1); // last batch size
    std::vector<neuron_state> res(config->n_outputs, {0, 0});
    double cost, batch_cost = -1, highest_cost, batch_prog;
    std::chrono::_V2::system_clock::time_point start, end;
    std::chrono::milliseconds duration;
    int progress_bar_size = 20;
    // set up before entering loop
    std::ios_base::fmtflags f( std::cout.flags() );
    std::cout << std::fixed << std::setprecision(2);
    for (size_t batch = 0; batch < batches; batch++) {
        batch_cost = 0;
        highest_cost = 0;
        batch_prog = 0;
        start = std::chrono::high_resolution_clock::now();
        for (size_t batch_c = 0; batch_c < total_splits; batch_c++) {
            for (size_t i = 0; i < batch_s; i++) {
                if (batch_c == total_splits-1 && i >= lbatch_size) {
                    break;
                }
                res = this->predict(inputs[batch_c*batch_s+i]);
                // error calculation
                cost = config->e_func.func(res, targets[batch_c*batch_s+i]);
                batch_cost += cost;
                if (highest_cost < cost) {
                    highest_cost = cost;
                }
                this->optimizer->step(targets[batch_c*batch_s+i]);
                batch_prog += 1.0 / (double) total_samples;
            }
            // after each sub batch update weights
            this->weights = this->s_weights;
            this->biases = this->s_biases;
            if (!config->verbose) continue;
            std::cout << "[";
            for (int c = 0; c < progress_bar_size; c++) {
                if (c < progress_bar_size*batch_prog)
                    std::cout << "=";
                else 
                    std::cout << "-";
            }
            std::cout << "] " << batch_prog*100 << "%, " << (batch_c)*batch_s << "/" << total_samples << "                              \r";
            std::cout.flush();
            
        }
        end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        // batch_cost /= total_samples;
        
        std::cout << "\nItr: " << iteration << ", " << duration.count() << "ms" << ", cost: " << batch_cost << ", highest: " << highest_cost << std::endl;
        iteration++;
        
    }

    // and unsetup to not accidentally mess with other stuff
    std::cout.flags(f);
    return batch_cost;
}

NeuralNetwork::GradientDescendOptim::~GradientDescendOptim() {
}

void NeuralNetwork::GradientDescendOptim::set_owner(std::shared_ptr<FFNeuralNetwork> net) {
    if (this->net_owner != nullptr) {
        throw std::runtime_error("set_owner called twice");
    }
    this->net_owner = net;
    std::shared_ptr<FFNNConfig> curr_config = net->get_config();
    int biggest_layer = 0;
    for (size_t i = 0; i < curr_config->n_layers.size() - 1; i++) {
        if (biggest_layer < curr_config->n_layers[i]) {
            biggest_layer = curr_config->n_layers[i];
        }
    }
    this->dLofa.resize(biggest_layer); // this will reserve extra space, with the goal of not having to create a vector each time
    this->temp_dLofa.resize(biggest_layer);
    this->daofz.resize(biggest_layer);
    for (size_t i = 0; i < this->daofz.size(); i++)
        this->daofz[i].resize(biggest_layer);
    this->dLofb.resize(biggest_layer);
    this->dLofw.resize(biggest_layer); // this one will be square matrix
    for (size_t i = 0; i < this->dLofw.size(); i++)
        this->dLofw[i].resize(biggest_layer);
}

void NeuralNetwork::GradientDescendOptim::step(std::vector<double> &targets) {
    if (!this->net_owner) {
        throw std::runtime_error("step: owner was not set");
    }
    gradient_descent(targets);
}

void NeuralNetwork::GradientDescendOptim::gradient_descent(std::vector<double> &targets) {
    std::shared_ptr<FFNNConfig> curr_conf = net_owner->get_config();
    size_t ll_index = net_owner->activs.size()-1;
    if (targets.size() != net_owner->activs[ll_index].size()) 
        throw std::runtime_error("gradient_descent: targets.size() != activs.size()");
    // setup dLofa to start loop 
    curr_conf->e_func.dfunc(net_owner->activs[ll_index], targets, this->dLofa);
    // now dLofa is the derivative of cost function

    for (size_t wl = curr_conf->n_layers.size()-1; wl > 0; wl--) { // iterate from end to beginning
        jacob_act_func* ptr = dynamic_cast<jacob_act_func*>(curr_conf->act_funcs[wl-1]);
        if (ptr != nullptr) { // special handling for jacobian matrix
            ptr->djfunc(net_owner->activs[wl], daofz);
            double sum;
            for (int i = 0; i < curr_conf->n_layers[wl]; i++) {
                sum = 0;
                for (int j = 0; j < curr_conf->n_layers[wl]; j++) {
                    sum += dLofa[j] * daofz[i][j];
                }
                dLofb[i] = sum;
            }
        } else { // normal handling
            curr_conf->act_funcs[wl-1]->dfunc(net_owner->activs[wl], daofz[0]); // [0] because we only use first row as it acts as a vector
            for (int i = 0; i < curr_conf->n_layers[wl]; i++) // we know that dLofa.size() == daofz.size()
                dLofb[i] = dLofa[i] * daofz[i][0]; // * 1; dzofb is always 1
        }
        // loop through all weights, ws goes up to wl-1 because of the way dLofw is setup and we up to wl
        for (int ws = 0; ws < curr_conf->n_layers[wl-1]; ws++) {
            for (int we = 0; we < curr_conf->n_layers[wl]; we++) {
                dLofw[ws][we] = dLofb[we] * net_owner->activs[wl-1][ws].result; // using dLofb as it is dLofa*daofz.
                                                                            // dzofw is the activ[wl-1]
            }
        }

        // gradient descent calculated, time to shift values
        shift_values(wl);

        temp_dLofa = dLofa;

        // finally set up dLofa for next iteration
        for (int i = 0; i < curr_conf->n_layers[wl-1]; i++) {
            double sum = 0;
            for (int j = 0; j < curr_conf->n_layers[wl]; j++) {
                sum += temp_dLofa[j] * (dLofb[j]/temp_dLofa[j]) * net_owner->weights[wl-1][i][j]; // deduce daofz from divison to prevent special cases for jacobian matrix
            }
            dLofa[i] = sum;
        }
    }
}

NeuralNetwork::BProp::BProp(double learning_rate, OptimizerFunctions::LearningRateOptimizerBase& lr_func) {
    this->learning_rate = learning_rate;
    this->lr_func = std::unique_ptr<OptimizerFunctions::LearningRateOptimizerBase>(lr_func.make_copy());
}

NeuralNetwork::BProp::~BProp() {
}

void NeuralNetwork::BProp::shift_values(int wl) {
    if (!this->net_owner) {
        throw std::runtime_error("set_owner was not called");
    }
    std::shared_ptr<FFNNConfig> curr_conf = net_owner->get_config();
    for (int n_i = 0; n_i < curr_conf->n_layers[wl]; n_i++) {
        net_owner->s_biases[wl][n_i] -= this->learning_rate*this->dLofb[n_i];
        for (int prev_n_i = 0; prev_n_i < curr_conf->n_layers[wl-1]; prev_n_i++) {
            net_owner->s_weights[wl-1][prev_n_i][n_i] -= this->learning_rate*this->dLofw[prev_n_i][n_i];
        }
    }
    this->learning_rate = this->lr_func.get()->step(this->learning_rate, net_owner->iteration);
}

NeuralNetwork::GradientDescendOptim* NeuralNetwork::BProp::make_copy() {
    NeuralNetwork::BProp* copy = new NeuralNetwork::BProp(this->learning_rate, *this->lr_func.get());
    return copy;
}

NeuralNetwork::Adam::Adam(double alpha, double beta1, double beta2, double elipson) {
    this->alpha = alpha;
    this->beta1 = beta1;
    this->beta2 = beta2;
    this->pow_beta1 = beta1;
    this->pow_beta2 = beta2;
    this->elipson = elipson;
}

NeuralNetwork::Adam::~Adam() {
}

void NeuralNetwork::Adam::set_owner(std::shared_ptr<FFNeuralNetwork> net) {
    GradientDescendOptim::set_owner(net);
    this->last_itr = 1;
    this->m_w = net->weights; // copy from weights
    this->v_w = net->weights; // to set it up automatically
    for (size_t i = 0; i < m_w.size(); i++) {
        for (size_t j = 0; j < m_w[i].size(); j++) {
            for (size_t k = 0; k < m_w[i][j].size(); k++) {
                this->m_w[i][j][k] = 0; // then reset the values
                this->v_w[i][j][k] = 0;
            }
        }
    }
    // same for bias
    this->m_b = net->biases;
    this->v_b = net->biases;
    for (size_t i = 0; i < m_b.size(); i++) {
        for (size_t j = 0; j < m_b[i].size(); j++) {
            this->m_b[i][j] = 0;
            this->v_b[i][j] = 0;
        }
    }
}

void NeuralNetwork::Adam::shift_values(int wl) {
    if (!this->net_owner) {
        throw std::runtime_error("set_owner was not called");
    }
    std::shared_ptr<FFNNConfig> curr_conf = net_owner->get_config();
    double mhat_b = 0;
    double vhat_b = 0;
    double mhat_w = 0;
    double vhat_w = 0;
    for (unsigned long long i = last_itr; i < net_owner->iteration+1; i++) {
        this->pow_beta1 *= beta1;
        this->pow_beta2 *= beta2;
        last_itr++;
    }
    for (int n_i = 0; n_i < curr_conf->n_layers[wl]; n_i++) {
        this->m_b[wl][n_i] = beta1 * m_b[wl][n_i] + (1.0 - beta1) * dLofb[n_i];
        this->v_b[wl][n_i] = beta2 * v_b[wl][n_i] + (1.0 - beta2) * dLofb[n_i] * dLofb[n_i];
        mhat_b = m_b[wl][n_i] / (1.0 - pow_beta1);
        vhat_b = v_b[wl][n_i] / (1.0 - pow_beta2);
        net_owner->s_biases[wl][n_i] = net_owner->s_biases[wl][n_i] - alpha * mhat_b / (std::sqrt(vhat_b) + elipson);
        for (int prev_n_i = 0; prev_n_i < curr_conf->n_layers[wl-1]; prev_n_i++) {
            this->m_w[wl-1][prev_n_i][n_i] = beta1 * m_w[wl-1][prev_n_i][n_i] + (1.0 - beta1) * dLofw[prev_n_i][n_i];
            this->v_w[wl-1][prev_n_i][n_i] = beta2 * v_w[wl-1][prev_n_i][n_i] + (1.0 - beta2) * dLofw[prev_n_i][n_i] * dLofw[prev_n_i][n_i];
            mhat_w = m_w[wl-1][prev_n_i][n_i] / (1.0 - pow_beta1);
            vhat_w = v_w[wl-1][prev_n_i][n_i] / (1.0 - pow_beta2);
            net_owner->s_weights[wl-1][prev_n_i][n_i] = net_owner->s_weights[wl-1][prev_n_i][n_i] - alpha * mhat_w / (std::sqrt(vhat_w) + elipson);
        }
    }
}

NeuralNetwork::GradientDescendOptim* NeuralNetwork::Adam::make_copy() {
    NeuralNetwork::Adam* copy = new NeuralNetwork::Adam(this->alpha, this->beta1, this->beta2, this->elipson);
    return copy;
}