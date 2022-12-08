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

int main() {
    std::vector<int> c1 = {-1,  1,  0, 
                            0,  1,  1, 
                           -1,  1,  0};
    std::ios_base::fmtflags f( std::cout.flags() );
    std::cout << possibleWins(c1, -1) << std::endl;
    std::cout << 324.533333333 << std::endl;
    
    std::cout << std::fixed << std::setprecision(5) << 5234.3432423523 << std::endl;
    std::cout.flags(f);
    std::cout << 324.533333333 << std::endl;
}