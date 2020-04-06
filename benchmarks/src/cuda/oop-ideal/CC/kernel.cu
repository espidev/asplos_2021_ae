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
	for (int i = in_start; i < in_end; i++) {
	    vertex[tid]->setInEdge(i - in_start, incol[i], 0);
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
	for (int i = out_start; i < out_end; i++) {
	    vertex[tid]->setOutEdge(vertex, tid, i - out_start, col[i], 0);
	}
    }
}

__global__ void ConnectedComponent(ChiVertex<int, int> **vertex, GraphChiContext* context) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < context->getNumVertices()) {
	int iteration = context->getNumIterations();
	int numEdges = vertex[tid]->numEdges();
	if (iteration == 0) {
	    vertex[tid]->setValue(vertex[tid]->getId());
	}
	int curMin = vertex[tid]->getValue();
        for(int i=0; i < numEdges; i++) {
            int nbLabel = ((Edge<float> *)vertex[tid]->edge(i))->getValueConcrete();
            if (iteration == 0) nbLabel = ((Edge<float> *)vertex[tid]->edge(i))->getVertexIdConcrete(); // Note!
            if (nbLabel < curMin) {
                curMin = nbLabel;
            }
	}

        /**
         * Set my new label
         */
        vertex[tid]->setValue(curMin);
        int label = curMin;

        /**
         * Broadcast my value to neighbors by writing the value to my edges.
         */
        if (iteration > 0) {
            for(int i=0; i < numEdges; i++) {
                if (((Edge<float> *)vertex[tid]->edge(i))->getValueConcrete() > label) {
                    ((Edge<float> *)vertex[tid]->edge(i))->setValueConcrete(label);
                }
            }
        } else {
            // Special case for first iteration to avoid overwriting
            for(int i=0; i < vertex[tid]->numOutEdges(); i++) {
                ((Edge<float> *)vertex[tid]->getOutEdge(i))->setValueConcrete(label);
            }
        }
	context->setNumIterations(context->getNumIterations() + 1);
    }
}

__global__ void copyBack(ChiVertex<int, int> **vertex, GraphChiContext* context,
	int *cc)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < context->getNumVertices()) {
        cc[tid] = vertex[tid]->getValue();
    }
}
