__global__ void initContext(GraphChiContext* context, int vertices, int edges) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid == 0) {
	context->setNumIterations(0);
	context->setNumVertices(vertices);
	context->setNumEdges(edges);
    }
}

__global__ void initObject(ChiVertex<int, int> **vertex, GraphChiContext* context,
	int* row, int* col, int* inrow, int* incol) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < context->getNumVertices()) {
	int out_start = row[tid];
	int out_end;
	if (tid + 1 < context->getNumVertices()) {
	    out_end = row[tid + 1];
	} else {
	    out_end = context->getNumEdges();
	}
	int in_start = inrow[tid];
	int in_end;
	if (tid + 1 < context->getNumVertices()) {
	    in_end = inrow[tid + 1];
	} else {
	    in_end = context->getNumEdges();
	}
	int indegree = in_end - in_start;
	int outdegree = out_end - out_start;
	vertex[tid] = new ChiVertex<int, int>(tid, indegree, outdegree);
	vertex[tid]->setValue(INT_MAX);
	for (int i = in_start; i < in_end; i++) {
	    vertex[tid]->setInEdge(i - in_start, incol[i], INT_MAX);
	}
	//for (int i = out_start; i < out_end; i++) {
	//    vertex[tid]->setOutEdge(vertex, tid, i - out_start, col[i], 0.0f);
	//}
    }
}

__global__ void initOutEdge(ChiVertex<int, int> **vertex, GraphChiContext* context,
	int* row, int* col) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < context->getNumVertices()) {
	int out_start = row[tid];
	int out_end;
	if (tid + 1 < context->getNumVertices()) {
	    out_end = row[tid + 1];
	} else {
	    out_end = context->getNumEdges();
	}
	//int in_start = inrow[tid];
	//int in_end;
	//if (tid + 1 < context->getNumVertices()) {
	//    in_end = inrow[tid + 1];
	//} else {
	//    in_end = context->getNumEdges();
	//}
	//int indegree = in_end - in_start;
	//int outdegree = out_end - out_start;
	//vertex[tid] = new ChiVertex<float, float>(tid, indegree, outdegree);
	//for (int i = in_start; i < in_end; i++) {
	//    vertex[tid]->setInEdge(i - in_start, incol[i], 0.0f);
	//}
	for (int i = out_start; i < out_end; i++) {
	    vertex[tid]->setOutEdge(vertex, tid, i - out_start, col[i], INT_MAX);
	}
    }
}

__global__ void BFS(ChiVertex<int, int> **vertex, GraphChiContext* context, int iteration) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < context->getNumVertices()) {
        if (iteration == 0) {
            if (tid == 0) {
                vertex[tid]->setValue(0);
                for (int i = 0; i < vertex[tid]->numOutEdges(); i++) {
                    vertex[tid]->getOutEdge(i)->setValue(1);
                }
            }
        } else {
            int curmin = vertex[tid]->getValue();
            for (int i = 0; i < vertex[tid]->numInEdges(); i++) {
                curmin = min(curmin, vertex[tid]->getInEdge(i)->getValue());
            }
            if (curmin < vertex[tid]->getValue()) {
                vertex[tid]->setValue(curmin);
                for (int i = 0; i < vertex[tid]->numOutEdges(); i++) {
                    if (vertex[tid]->getOutEdge(i)->getValue() > curmin + 1){
                        vertex[tid]->getOutEdge(i)->setValue(curmin + 1);
                    }
                }
            }
        }
    }
}

__global__ void copyBack(ChiVertex<int, int> **vertex, GraphChiContext* context,
	int *index)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < context->getNumVertices()) {
        index[tid] = vertex[tid]->getValue();
    }
}
