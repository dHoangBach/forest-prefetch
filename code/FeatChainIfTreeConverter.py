from ForestConverter import TreeConverter
import numpy as np
from functools import reduce
import heapq
import gc

class FeatChainIfTreeConverter(TreeConverter):
    def __init__(self, dim, namespace, featureType):
        super().__init__(dim, namespace, featureType)

    def getImplementation(self, treeID, head, level = 1):
        code = ""
        tabs = "".join(['\t' for i in range(level)])

        if head.prediction is not None:
            return tabs + "return " + str(int(np.argmax(head.prediction))) + ";\n" ;
        else:

# OWN CODING #######################################################################################################################

                code += tabs + "if(pX[" + str(head.feature) + "] <= " + str(head.split) + "){\n"
                code += self.getImplementation(treeID, head.leftChild, level + 1)
                if head.leftChild.feature is not None:
                    code += tabs + """     __builtin_prefetch ( &pX[{tree}] );\n""".replace("{tree}", str(head.leftChild.feature))
                code += self.chainProbChild(head, (head.leftChild), 3, "", tabs)
                code += tabs + "} else {\n"     # else part
                code += self.getImplementation(treeID, head.rightChild, level + 1)
                if head.rightChild.feature is not None:
                    code += tabs + """     __builtin_prefetch ( &pX[{tree}] );\n""".replace("{tree}", str(head.rightChild.feature))
                code += self.chainProbChild(head, (head.rightChild), 3, "", tabs)
                code += tabs + "}\n"
        return code

####################################################################################################################################

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

# OWN CODING #######################################################################################################################

    def chainProbChild(self, head, node, counter, code, tabs):
        if counter > 0:
            if node.probLeft is not None:
                if node.probRight is not None:
                    if (float(node.probLeft) < float(node.probRight)):
                        if node.rightChild.feature is not None:
                            code += tabs + """     __builtin_prefetch ( &pX[{tree}] );\n""".replace("{tree}", str(node.rightChild.feature))
                        return self.chainProbChild(head, node.rightChild, counter-1, code, tabs)
                    else:
                        if node.leftChild.feature is not None:
                            code += tabs + """     __builtin_prefetch ( &pX[{tree}] );\n""".replace("{tree}", str(node.leftChild.feature))
                        return self.chainProbChild(head, node.leftChild, counter-1, code, tabs)
                else:
                    if node.leftChild.feature is not None:
                        code += tabs + """     __builtin_prefetch ( &pX[{tree}] );\n""".replace("{tree}", str(node.leftChild.feature))
                    return self.chainProbChild(head, node.leftChild, counter-1, code, tabs)
            else:
                if node.probRight is not None:
                    if node.rightChild.feature is not None:
                        code += tabs + """     __builtin_prefetch ( &pX[{tree}] );\n""".replace("{tree}", str(node.rightChild.feature))
                    return self.chainProbChild(head, node.rightChild, counter-1, code, tabs)
                else:
                    if head.feature is not None:
                        code += tabs + """     __builtin_prefetch ( &pX[{tree}] );\n""".replace("{tree}", str(head.feature))
                    return self.chainProbChild(head, head, counter-1, code, tabs)
        else:
            return code

####################################################################################################################################

