#include <iostream>
using namespace std;

struct Node {
    int node_id;
    string type;
    double bias;
};

struct Link_Id {
    int in_id;
    int out_id;
};

struct Conn {
    int inno_no;
    bool enable;
    double weight;
    Link_Id link;
};

int main() {
    cout << "Hello World";
    return 0;
}