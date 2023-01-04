#include <neuralnetwork.h>
#include <functions.h>
#include <vector>
#include <iostream>
#include <string>
#include <fstream>
#include <string>
#include <thread>
#include <profiler.h>

std::vector<std::vector<double>> train_X;
std::vector<std::vector<double>> train_Y;

std::vector<std::vector<double>> mnist_train_X;
std::vector<std::vector<double>> mnist_train_Y;

std::vector<std::vector<double>> smol_train_X;
std::vector<std::vector<double>> smol_train_Y;

std::vector lays = {20, 10};
std::vector<act_func*> a_f = {
    &ActivationFunctions::relu,
    &ActivationFunctions::relu,
    &ActivationFunctions::softmax
};

int reverse_int(int i) {
    unsigned char c1, c2, c3, c4;

    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;

    return ((int) c1 << 24) + ((int) c2 << 16) + ((int) c3 << 8) + c4;
}

// stolen from: https://stackoverflow.com/questions/8286668/how-to-read-mnist-data-in-c
void read_mnist_train() {
    std::string target_file = "./resources/train-images-idx3-ubyte";
    std::ifstream file (target_file, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("cannot open target file: " + target_file);
    }

    int magic_number = 0;
    file.read((char*) &magic_number, sizeof(magic_number)); 
    magic_number = reverse_int(magic_number);
    if (magic_number != 2051) {
        throw std::runtime_error("invalid magic number for file: " + target_file);
    }

    int number_of_images = 0;
    int n_rows = 0;
    int n_cols = 0;

    file.read((char*) &number_of_images, sizeof(number_of_images));
    number_of_images = reverse_int(number_of_images);
    mnist_train_X.resize(number_of_images);

    file.read((char*) &n_rows, sizeof(n_rows));
    n_rows = reverse_int(n_rows);

    file.read((char*) &n_cols, sizeof(n_cols));
    n_cols = reverse_int(n_cols);

    for(int i = 0; i < number_of_images; i++) {
        mnist_train_X[i].resize(n_cols*n_rows);
        for(int j = 0; j < n_rows*n_cols; j++) {
            unsigned char temp = 0;
            file.read((char*) &temp, sizeof(temp));
            mnist_train_X[i][j] = ((double)temp)/128-1; // this transforms 0 - 255 to -1 - 1
        }
    }
}

// stolen from: https://stackoverflow.com/questions/8286668/how-to-read-mnist-data-in-c
void read_mnist_train_lab() {
    std::string target_file = "./resources/train-labels-idx1-ubyte";
    std::ifstream file (target_file, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("cannot open target file: " + target_file);
    }
    int magic_number = 0;
    file.read((char*) &magic_number, sizeof(magic_number)); 
    magic_number = reverse_int(magic_number);
    if (magic_number != 2049) {
        throw std::runtime_error("invalid magic number for file: " + target_file);
    }
    int number_of_labels = 0;
    file.read((char*) &number_of_labels, sizeof(number_of_labels));
    number_of_labels = reverse_int(number_of_labels);
    mnist_train_Y.resize(number_of_labels);
    for(int i = 0; i < number_of_labels; i++) {
        unsigned char temp = 0;
        file.read((char*) &temp, sizeof(temp));
        mnist_train_Y[i].resize(10);
        for (int j = 0; j < 10; j++) {
            mnist_train_Y[i][j] = (int) temp == j ? 1 : 0;
        }
    }
    
}

double target_func(double x, double y) {
    return ((x*x+y*y)/100+std::sin(x)*10+std::sin(y)*10)/10;
    // return x*x+y*y;
}

void run_test(std::pair<int, double>* out, int st, int end) {
    auto init_f = std::make_shared<InitialitzationFunctions::HeInit>(41);
    NeuralNetwork::FFNNConfig config(28*28, 10, lays, a_f, init_f, ErrorFunctions::cross_entropy); // TODO add softmax error calc error
    config.verbose = false;
    // net config can be recycled
    double lowest_cost = INFINITY;
    int seed = 0;
    int best_seed = 0;

    for (int i = st; i < end; i++) {
        seed = i;
        init_f->set_seed(seed);
        // OptimizerFunctions::LRDecay *dec = new OptimizerFunctions::LRDecay(0.000005);
        // NeuralNetwork::BProp *bprop = new NeuralNetwork::BProp(1e-2, dec);
        // optimizer cann not
        NeuralNetwork::Adam adam(0.002, 0.9, 0.999, 1e-8);
        std::shared_ptr<NeuralNetwork::FFNeuralNetwork> net = NeuralNetwork::FFNeuralNetwork::create(config, adam);
        
        double cost = net->fit(mnist_train_X, mnist_train_Y, 32, 1);
        std::cout << "^Seed: " << seed << std::endl;
        if (cost < lowest_cost) {
            lowest_cost = cost;
            best_seed = seed;
        }
    }
    *out = {best_seed, lowest_cost};
}

int get_good_seed(int seeds_to_test, int threadc) {
    std::vector<std::thread> threads;
    std::pair<int, double> *res_threads = new std::pair<int, double>[threadc];
    threadc = (seeds_to_test<threadc) ? seeds_to_test : threadc;
    int step = seeds_to_test/threadc;
    int st = 0, end = 0, curr = 0;
    for (int i = 0; i < threadc; i++) {
        curr = step;
        if (i < step) {
            curr++;
        }
        end += curr;
        threads.push_back(std::thread(run_test, res_threads+i, st, end));
        st += curr;
    }
    for (size_t i = 0; i < threads.size(); i++) {
        threads[i].join();
    }
    
    int best_seed = INFINITY;
    double lowest_cost = INFINITY;
    for (int i = 0; i < threadc; i++) {
        if (res_threads[i].second < lowest_cost) {
            lowest_cost = res_threads[i].second;
            best_seed = res_threads[i].first;
        }
    }
    delete[] res_threads;
    if (best_seed == INFINITY) throw std::runtime_error("bad");
    std::cout << "Seed chosen: " << best_seed << std::endl;
    return best_seed;
}

int main() {
    Profiler::put_time("load_mnist");
    read_mnist_train();
    read_mnist_train_lab();
    Profiler::pop_time();
    
    // std::vector<act_func> ptr = {ActivationFunctions::softmax};
    // jacob_act_func* jptr = dynamic_cast<jacob_act_func*>(&ptr[0]);
    // jacob_act_func* jaf = dynamic_cast<jacob_act_func*>(a_f[2]);
    
    // size_t range = 50;
    // for (size_t x = 0; x < range; x++) {
    //     for (size_t y = 0; y < range; y++) {
    //         double p_x = (x-range/2.0)/range;
    //         double p_y = (y-range/2.0)/range;
    //         train_X.push_back({p_x, p_y});
    //         train_Y.push_back({target_func(p_x, p_y)});
    //     }
    // }
    // mnist testing

    for (size_t i = 0; i < mnist_train_X.size()/1000; i++) {
        smol_train_X.push_back(mnist_train_X[i]);
        smol_train_Y.push_back(mnist_train_Y[i]);
    }
    // int best_seed = get_good_seed(10*5, 10);
    // InitialitzationFunctions::HeInit init_f(best_seed);
    Profiler::put_time("create nn");

    Profiler::put_time("he_init");
    auto init_f = std::make_shared<InitialitzationFunctions::HeInit>(41);
    Profiler::pop_time();

    Profiler::put_time("nnconfig");
    NeuralNetwork::FFNNConfig config(28*28, 10, lays, a_f, init_f, ErrorFunctions::cross_entropy); 
    Profiler::pop_time();
    
    Profiler::put_time("adam");
    NeuralNetwork::Adam adam(0.002, 0.9, 0.999, 1e-8);
    Profiler::pop_time();

    Profiler::put_time("net");
    std::shared_ptr<NeuralNetwork::FFNeuralNetwork> net = NeuralNetwork::FFNeuralNetwork::create(config, adam);
    Profiler::pop_time();

    Profiler::pop_time();

    Profiler::put_time("train");
    net->fit(mnist_train_X, mnist_train_Y, 32, 3);
    Profiler::pop_time();
    
}