from ForestConverter import TreeConverter
import numpy as np
from functools import reduce
import heapq
import gc

class LabelSkipIfTreeConverter(TreeConverter):
    def __init__(self, dim, namespace, featureType):
        super().__init__(dim, namespace, featureType)

    def getImplementation(self, treeID, head, level = 1, start=0):
        code = ""
        tabs = "".join(['\t' for i in range(level)])

        if head.prediction is not None:
            return tabs + "    return " + str(int(np.argmax(head.prediction))) + ";\n" ;
        else:

# OWN CODING #######################################################################################################################

            if (start is 0): # for root
                code += "label_{number}:\n".replace("{number}", str(head))

            code += tabs + "    if(pX[" + str(head.feature) + "] <= " + str(head.split) + "){\n"
            code += "label_{number}:\n".replace("{number}", str(head.leftChild))                
            code += self.getImplementation(treeID, head.leftChild, level + 1, 1)
            code += tabs + self.skipProbChild(head, (head.leftChild), 2)
                # else
            code += tabs + "    } else {\n"
            code += "label_{number}:\n".replace("{number}", str(head.rightChild))                
            code += self.getImplementation(treeID, head.rightChild, level + 1, 1)
            code += tabs + self.skipProbChild(head, (head.rightChild), 2)
            code += tabs + "    }\n"
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

    def skipProbChild(self, head, node, counter):
        if counter > 0:
            if node.probLeft is not None:
                if node.probRight is not None:
                    if (float(node.probLeft) < float(node.probRight)):
                         return self.skipProbChild(head, node.rightChild, counter-1)
                    else:
                         return self.skipProbChild(head, node.leftChild, counter-1)
                else:
                     return self.skipProbChild(head, node.leftChild, counter-1)
            else:
                if node.probRight is not None:
                     return self.skipProbChild(head, node.rightChild, counter-1)
                else:
                     return self.skipProbChild(head, head, counter-1)
        else:
            return """     __builtin_prefetch ( &&label_{number} );\n""".replace("{number}", str(node))

####################################################################################################################################