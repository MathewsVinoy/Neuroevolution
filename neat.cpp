#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
using namespace std;

struct Node
{
    int id;
    string type;
    double bias;
};

struct Link_Id
{
    int input_Id;
    int output_Id;
};

struct Conn
{
    int inno_no;
    Link_Id link_id;
    double weight;
    bool is_enable;
};

struct Genome
{
    int id;
    double fitness;
    vector<Node> nodes;
    vector<Conn> conns;
};

int innovation_tracker()
{
    return 0;
}

void mutate(Genome &genome)
{
    double random_no = (double)rand() / RAND_MAX; // Generate random number between 0 and 1

    if (random_no < 0.9)
    {
        // Add node mutation logic
        Conn old_conn = genome.conns[rand() % genome.conns.size()];
        old_conn.is_enable = false;
        Node new_node;
        new_node.id = genome.nodes.size();
        new_node.type = "HIDDEN";
        genome.nodes.push_back(new_node);
        Conn conn1, conn2;
        Link_Id link1, link2;
        link1.input_Id = old_conn.link_id.input_Id;
        link1.output_Id = new_node.id;
        link2.input_Id = new_node.id;
        link2.output_Id = old_conn.link_id.output_Id;
        conn1.inno_no = innovation_tracker();
        conn1.link_id = link1;
        conn1.weight = 1.0;
        conn1.is_enable = true;
        conn2.inno_no = innovation_tracker();
        conn2.link_id = link2;
        conn2.weight = old_conn.weight;
        conn2.is_enable = true;
    }
    else if (random_no < 0.5)
    {
        // Mutation case 2
    }
    else
    {
        // Mutation case 3
    }
}

int main()
{
    srand(time(0));

    Node n1;
    n1.id = 1;
    n1.type = "INPUT";
    n1.bias = 1.1;

    cout << n1.id;
    return 0;
}