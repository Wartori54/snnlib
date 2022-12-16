#include <iostream>
#include <vector>
#include <iomanip>
#include <ios>

std::vector<std::vector<int>> possible_cases = {
    {0, 1, 2},
    {3, 4, 5},
    {6, 7, 8},
    {0, 3, 6},
    {1, 4, 7},
    {2, 5, 8},
    {0, 4, 8},
    {2, 4, 6}
};

int possibleWins(std::vector<int> pieces, int player) {
    if (pieces.size() != 9) return 0;
    int count = 0;
    for (int c = 0; c < possible_cases.size(); c++) {
        count++;
        for (int i = 0; i < possible_cases[c].size(); i++) {
            if (pieces[possible_cases[c][i]] == -player) {
                count--;
                break;
            }
        }
    }
    return count;
}

void statictest(std::vector<double>& vec) {
    static std::vector<double> st(vec.size(), 0);
    std::cout << "st size: " << st.size() << std::endl;
}

struct A {
    int n1;
    float f1;
    A(int n, float f) : n1(n), f1(f) {};
    virtual ~A() {};
};

struct B : public A {
    int n2;
    B(int n) : A(-1, 100), n2(n) {};
};

int main() {
    // std::vector<double> testv1(5, 0);
    // std::vector<double> testv(10, 0);

    // statictest(testv1);

    // statictest(testv);

    // std::vector<int> c1 = {-1,  1,  0, 
    //                         0,  1,  1, 
    //                        -1,  1,  0};
    // std::ios_base::fmtflags f( std::cout.flags() );
    // std::cout << possibleWins(c1, -1) << std::endl;
    // std::cout << 324.533333333 << std::endl;
    
    // std::cout << std::fixed << std::setprecision(5) << 5234.3432423523 << std::endl;
    // std::cout.flags(f);
    // std::cout << 324.533333333 << std::endl;

    A a1(1, 3);
    B b1(4);

    A* a2 = &b1;

    B* b2 = dynamic_cast<B*>(a2);

    if (__cplusplus == 201703L) std::cout << "C++17\n";
    else if (__cplusplus == 201402L) std::cout << "C++14\n";
    else if (__cplusplus == 201103L) std::cout << "C++11\n";
    else if (__cplusplus == 199711L) std::cout << "C++98\n";
    else std::cout << "pre-standard C++\n";
    std::cin.get();
}