#define MAX_LINE_LENGTH 500000

class Graph
{
    unsigned int num_verts;
    unsigned int num_edges;
    unsigned int adj_list_length;
    unsigned int *edge_offsets;
    unsigned int *edge_list;
    unsigned int *edge_costs;
    unsigned int max_degree;
    int graph_type;

    bool if_delete_arrays;

    void SetAllCosts(unsigned int c);
    public:
    Graph();
    ~Graph();
    void LoadMetisGraph(const char *filename);
    void SaveMetisGraph(const char *filename);
    unsigned int GetNumVertices();
    unsigned int GetNumEdges();
    unsigned int GetMaxDegree();

    unsigned int *GetEdgeOffsets();
    unsigned int *GetEdgeList();
    unsigned int *GetEdgeCosts();

    unsigned int **GetEdgeOffsetsPtr();
    unsigned int **GetEdgeListPtr();
    unsigned int **GetEdgeCostsPtr();

    unsigned int *GetVertexLengths(unsigned int *cost,unsigned int source);
    int GetMetisGraphType();
    unsigned int GetAdjacencyListLength();
    void GenerateSimpleKWayGraph(unsigned int verts,unsigned int degree);
};
