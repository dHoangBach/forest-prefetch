from ForestConverter import TreeConverter
import numpy as np
from functools import reduce
import heapq
import gc
#import objgraph

class DoubleIfTreeConverter(TreeConverter):
    def __init__(self, dim, namespace, featureType):
        super().__init__(dim, namespace, featureType)

    def getImplementation(self, treeID, head, level = 1):

        code = ""
        tabs = "".join(['\t' for i in range(level)])

        if head.prediction is not None:
            return tabs + "return " + str(int(np.argmax(head.prediction))) + ";\n" ;
        else:
                code += tabs + "if(pX[" + str(head.feature) + "] <= " + str(head.split) + "){\n"    # Condition feature <= split
                code += self.getImplementation(treeID, head.leftChild, level + 1)   # Insert leftChild, prefetch
                code += tabs + """     __builtin_prefetch ( &pX[{tree}] );\n""".replace("{tree}", str(head.leftChild))
                code += tabs + self.getProbChild(head, (head.leftChild))
                code += tabs + "} else {\n"     # else part
                code += self.getImplementation(treeID, head.rightChild, level + 1)  # Insert rightChild, prefetch
                code += tabs + """     __builtin_prefetch ( &pX[{tree}] );\n""".replace("{tree}", str(head.rightChild))
                code += tabs + self.getProbChild(head, (head.rightChild))
                code += tabs + "}\n"
        return code

    def getCode(self, tree, treeID, numClasses):

        featureType = self.getFeatureType()
        cppCode = "inline unsigned int {namespace}_predict{treeID}({feature_t} const pX[{dim}]){\n" \
                                .replace("{treeID}", str(treeID)) \
                                .replace("{dim}", str(self.dim)) \
                                .replace("{namespace}", self.namespace) \
                                .replace("{feature_t}", featureType) \
                                .replace("{numClasses}", str(numClasses))

        cppCode += self.getImplementation(treeID, tree.head)
        cppCode += "}\n"

        headerCode = "inline unsigned int {namespace}_predict{treeID}({feature_t} const pX[{dim}]);\n" \
                                        .replace("{treeID}", str(treeID)) \
                                        .replace("{dim}", str(self.dim)) \
                                        .replace("{namespace}", self.namespace) \
                                        .replace("{feature_t}", featureType) \
                                        .replace("{numClasses}", str(numClasses))


        return headerCode, cppCode

    def getProbChild(self, head, node):
        if node.probLeft is not None:
                if node.probRight is not None:
                    if (float(node.probLeft) < float(node.probRight)):
                         return """     __builtin_prefetch ( &pX[{tree}] );\n""".replace("{tree}", str(node.rightChild))
                    else:
                         return """     __builtin_prefetch ( &pX[{tree}] );\n""".replace("{tree}", str(node.leftChild))
                else:
                     return """     __builtin_prefetch ( &pX[{tree}] );\n""".replace("{tree}", str(node.leftChild))
        else:
                if node.probRight is not None:
                     return """     __builtin_prefetch ( &pX[{tree}] );\n""".replace("{tree}", str(node.rightChild))
                else:
                     return"""     __builtin_prefetch ( &pX[{tree}] );\n""".replace("{tree}", str(head))

