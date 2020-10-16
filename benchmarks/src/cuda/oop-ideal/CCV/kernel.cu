__global__ void initContext(GraphChiContext* context, int vertices, int edges) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid == 0) {
	context->setNumIterations(0);
	context->setNumVertices(vertices);
	context->setNumEdges(edges);
    }
}

__global__ void initObject(VirtVertex<int, int> **vertex, GraphChiContext* context,
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

__global__ void initOutEdge(VirtVertex<int, int> **vertex, GraphChiContext* context,
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

__global__ void ConnectedComponent(VirtVertex<int, int> **vertex, GraphChiContext* context, int iteration) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < context->getNumVertices()) {
        int numEdges;
        numEdges = ((ChiVertex<int, int> *)vertex[tid])->numEdgesConcrete();
        if (iteration == 0) {
            int vid = ((ChiVertex<int, int> *)vertex[tid])->getIdConcrete();
            ((ChiVertex<int, int> *)vertex[tid])->setValueConcrete(vid);
        }
        int curMin;
        curMin = ((ChiVertex<int, int> *)vertex[tid])->getValueConcrete();
        for(int i=0; i < numEdges; i++) {
            ChiEdge<int> * edge;
            edge = ((ChiVertex<int, int> *)vertex[tid])->edgeConcrete(i); 
            int nbLabel;
            nbLabel = ((Edge<int> *)edge)->getValueConcrete();
            if (iteration == 0) {
                nbLabel = ((Edge<int> *)edge)->getVertexIdConcrete(); // Note!
            }
            if (nbLabel < curMin) {
                curMin = nbLabel;
            }
        }

        /**
         * Set my new label
         */
        ((ChiVertex<int, int> *)vertex[tid])->setValue(curMin);
        int label = curMin;

        /**
         * Broadcast my value to neighbors by writing the value to my edges.
         */
        if (iteration > 0) {
            for(int i=0; i < numEdges; i++) {
                ChiEdge<int> * edge;
                edge = ((ChiVertex<int, int> *)vertex[tid])->edgeConcrete(i);
                int edgeValue;
                edgeValue = ((Edge<int> *)edge)->getValueConcrete();
                if (edgeValue > label) {
                    ((Edge<int> *)edge)->setValueConcrete(label);
                }
            }
        } else {
            // Special case for first iteration to avoid overwriting
            int numOutEdge;
            numOutEdge = ((ChiVertex<int, int> *)vertex[tid])->numOutEdgesConcrete();
            for(int i=0; i < numOutEdge; i++) {
                ChiEdge<int> * outEdge;
                outEdge = ((ChiVertex<int, int> *)vertex[tid])->getOutEdgeConcrete(i);
                ((Edge<int> *)outEdge)->setValueConcrete(label);
            }
        }
    }
}

__global__ void copyBack(VirtVertex<int, int> **vertex, GraphChiContext* context,
	int *cc)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < context->getNumVertices()) {
        cc[tid] = vertex[tid]->getValue();
    }
}
