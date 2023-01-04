#pragma once
#include <chrono>
#include <vector>
#include <string>
#include <filesystem>
#include <linkedlist>

namespace Profiler {
    typedef std::chrono::high_resolution_clock _clock;
    typedef std::chrono::microseconds time_resolution;

    extern std::string time_resolution_label;

    inline std::chrono::time_point<_clock> now();

    void save_data();

    extern std::string out_file;

    class TimeTree : public std::enable_shared_from_this<TimeTree> {
    public:
        std::weak_ptr<TimeTree> parent;
        Lists::LinkedList<std::shared_ptr<TimeTree>> childs;
    private:
        std::chrono::time_point<_clock> start;
        std::chrono::time_point<_clock> end;
        std::string name;
    public:
        TimeTree(std::string n);
        void end_time();
        void add_child(std::shared_ptr<TimeTree> &c);
        void write_data(std::string& buf);
    };

    extern std::shared_ptr<Profiler::TimeTree> tree_base;
    extern std::shared_ptr<Profiler::TimeTree> curr_node;



    void put_time(std::string name);
    void pop_time();

    void analyze_node(std::string& buf, std::shared_ptr<Profiler::TimeTree> node, size_t depth);

    const std::string get_time_string();
    std::filesystem::path get_out_file_path();
}
