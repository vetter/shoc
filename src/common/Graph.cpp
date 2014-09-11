#include <cstdio>
#include <cstdlib>
#include <cctype>
#include <cassert>
#include <cstring>
#include <iostream>
#include <limits.h>
#include <math.h>
#include <list>

#include "Graph.h"

Graph::Graph()
{
    num_verts=0;
    num_edges=0;
    max_degree=0;
    adj_list_length=0;
    edge_offsets=NULL;
    edge_list=NULL;
    edge_costs=NULL;
    graph_type=-1;
    if_delete_arrays=false;
}

Graph::~Graph()
{
    if(if_delete_arrays)
    {
        delete[] edge_offsets;
        delete[] edge_list;
        if(graph_type==1)
            delete[] edge_costs;
    }
}


// ****************************************************************************
//  Method:  Graph::LoadMetisGraph
//
//  Purpose:
//      Loads a graph from METIS file format.
//
//  Arguments:
//    filename: file name of the graph to load
//
//  Programmer:  Aditya Sarwade
//  Creation:    June 16, 2011
//
//  Modifications:
//
// ****************************************************************************
void Graph::LoadMetisGraph(const char *filename)
{

	FILE *fp=fopen(filename,"r");
    assert(fp);
	char charBuf[MAX_LINE_LENGTH];
	const char delimiters[]=" \n";
	char *temp_token=NULL;

    while(1)
    {
        fgets(charBuf,MAX_LINE_LENGTH,fp);
        temp_token = strtok (charBuf, delimiters);

        if(temp_token==NULL)
            continue;

        else if(temp_token[0]=='%')
            continue;

        else
            break;

    }

    assert(isdigit(temp_token[0]));
	num_verts  = atoi(temp_token);
	temp_token = strtok (NULL, delimiters);
    assert(isdigit(temp_token[0]));
	num_edges=atoi(temp_token);
	temp_token = strtok (NULL, delimiters);
    if(temp_token==NULL)
    {
        graph_type = 0;
    }
    else
    {
        assert(isdigit(temp_token[0]));
        graph_type=atoi(temp_token);
        if(graph_type!=0 && graph_type!=1 && graph_type!=100)
        {
            std::cout<<"\nSupported metis graph types are 0 and 1";
            return;
        }
    }

    if(edge_offsets==NULL)
    {
        if_delete_arrays=true;
        edge_offsets=new unsigned int[num_verts+1];
        edge_list=new unsigned int[num_edges*2];
        if(graph_type == 1)
            edge_costs=new unsigned int[num_edges*2];
    }

    unsigned int cost=0;
	unsigned int offset=0;
	for(int index=1;index<=num_verts;index++)
	{
		fgets(charBuf,MAX_LINE_LENGTH,fp);

		temp_token=strtok(charBuf,delimiters);

		if(temp_token==NULL)
		{
			edge_offsets[index-1]=offset;
			continue;
		}
        if(temp_token[0]=='%')
        {
            continue;
        }
        assert(isdigit(temp_token[0]));

		unsigned int vert=atoi(temp_token);
		edge_offsets[index-1]=offset;
		edge_list[offset]=vert-1;

        if(graph_type==1)
		{
            temp_token=strtok(NULL,delimiters);
            assert(temp_token);
            assert(isdigit(temp_token[0]));
		    cost=atoi(temp_token);
            edge_costs[offset]=cost;
        }

		//temp_value=(index-1)*(num_verts)+(vert-1);
		offset++;
		while((temp_token=(strtok(NULL,delimiters))))
		{
            assert(isdigit(temp_token[0]));
			vert=atoi(temp_token);
			//temp_value=(index-1)*(num_verts)+(vert-1);
			edge_list[offset]=vert-1;

            if(graph_type==1)
            {
                temp_token=strtok(NULL,delimiters);
                assert(temp_token);
                assert(isdigit(temp_token[0]));
                cost=atoi(temp_token);
                edge_costs[offset]=cost;
            }

			offset++;
		}
        if(max_degree < offset-edge_offsets[index-1])
            max_degree=offset-edge_offsets[index-1];
	}

    adj_list_length=offset;

	//Add length of the adjacency list to last position
	edge_offsets[num_verts]=offset;
    adj_list_length=offset;
    fclose(fp);
}

// ****************************************************************************
//  Method:  Graph::SaveMetisGraph
//
//  Purpose:
//      Saves the graph in METIS file format.
//
//  Arguments:
//    filename: path to save the graph
//
//  Programmer:  Aditya Sarwade
//  Creation:    June 16, 2011
//
//  Modifications:
//
// ****************************************************************************
void Graph::SaveMetisGraph(const char *filename)
{
    FILE *fp=fopen(filename,"w");
    assert(fp);

    fprintf(fp,"%u %u",num_verts,num_edges);
    if(graph_type!=0)
        fprintf(fp," %d",graph_type);
    fprintf(fp,"\n");

    for(int i=0;i<num_verts;i++)
    {
        unsigned int offset=edge_offsets[i];
        unsigned int next  =edge_offsets[i+1];
        while(offset<next)
        {
            fprintf(fp,"%u ",edge_list[offset]+1);
            if(graph_type==1)
            {
                fprintf(fp,"%u ",edge_costs[offset]);
            }
            offset++;
        }
        fprintf(fp,"\n");
    }
    fclose(fp);
}

unsigned int Graph::GetNumVertices()
{
    return num_verts;
}

unsigned int Graph::GetNumEdges()
{
    return num_edges;
}

unsigned int Graph::GetMaxDegree()
{
    return max_degree;
}

unsigned int *Graph::GetEdgeOffsets()
{
    return edge_offsets;
}

unsigned int *Graph::GetEdgeList()
{
    return edge_list;
}

unsigned int *Graph::GetEdgeCosts()
{
    return edge_costs;
}

unsigned int **Graph::GetEdgeOffsetsPtr()
{
    return &edge_offsets;
}

unsigned int **Graph::GetEdgeListPtr()
{
    return &edge_list;
}

unsigned int **Graph::GetEdgeCostsPtr()
{
    return &edge_costs;
}

int Graph::GetMetisGraphType()
{
    return graph_type;
}

unsigned int Graph::GetAdjacencyListLength()
{
    return adj_list_length;
}

// ****************************************************************************
//  Method:  Graph::GenerateSimpleKWayGraph
//
//  Purpose:
//      Generates a simple k-way tree from specified number of nodes and degree
//
//  Arguments:
//    verts: number of vertices in the graph
//    degree: specify k for k-way tree
//
//  Programmer:  Aditya Sarwade
//  Creation:    June 16, 2011
//
//  Modifications:
//
// ****************************************************************************
void Graph::GenerateSimpleKWayGraph(
    unsigned int verts,
    unsigned int degree)
{
	unsigned int index=0;
	unsigned int offset=0,j;
	unsigned int temp;

    if(edge_offsets==NULL)
    {
        if_delete_arrays=true;
        edge_offsets=new unsigned int[verts+1];
        edge_list=new unsigned int[verts*(degree+1)];
    }

	for(index=0;index<verts;index++)
	{
		edge_offsets[index]=offset;
		for(j=0;j<degree;j++)
		{
			temp=index*degree+(j+1);
			if(temp<verts)
			{
				edge_list[offset]=temp;
				offset++;
			}
		}
		if(index!=0)
		{
			edge_list[offset]=(unsigned int)floor(
					(float)(index-1)/(float)degree);
			offset++;
		}
	}

	//Add length of the adjacency list to last position
	edge_offsets[verts]=offset;

    adj_list_length=offset;
    num_edges=offset/2;
    num_verts=verts;
    graph_type=0;

    max_degree = degree + 1;
}

// ****************************************************************************
//  Method:  Graph::GetVertexLengths
//
//  Purpose:
//      Calculates the path lengths of each vertex from a specified source vetex
//
//  Arguments:
//    cost: array to return the path lengths for each vertex.
//    source: source vertex to calculate path lengths from.
//
//  Programmer:  Aditya Sarwade
//  Creation:    June 16, 2011
//
//  Modifications:
//
// ****************************************************************************
unsigned int * Graph::GetVertexLengths(
		unsigned int *cost,
		unsigned int source)
{
	//BFS uses Queue data structure
	for(int i=0;i<num_verts;i++)
		cost[i]=UINT_MAX;

	cost[source]=0;
	unsigned int nid;
	unsigned int next,offset;
	int n;
	unsigned int num_verts_visited=0;
	std::list<unsigned int> q;
	n=q.size();
	q.push_back(source);
	while(!q.empty())
	{
		n=q.front();
		num_verts_visited++;
		q.pop_front();
		offset=edge_offsets[n];
		next=edge_offsets[n+1];
		while(offset<next)
		{
			nid=edge_list[offset];
			offset++;
			if(cost[nid]>cost[n]+1)
			{
				cost[nid]=cost[n]+1;
				q.push_back(nid);
			}
		}
	}
	return cost;
}
