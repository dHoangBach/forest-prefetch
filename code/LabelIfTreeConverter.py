from ForestConverter import TreeConverter
import numpy as np
from functools import reduce
import heapq
import gc
#import objgraph

class LabelIfTreeConverter(TreeConverter):
    def __init__(self, dim, namespace, featureType):
        super().__init__(dim, namespace, featureType)

    def getImplementation(self, treeID, head, level = 1):
        """ Generate the actual if-else implementation for a given node

        Args:
            treeID (TYPE): The id of this tree (in case we are dealing with a forest)
            head (TYPE): The current node to generate an if-else structure for.
            level (int, optional): The intendation level of the generated code for easier
                                                        reading of the generated code

        Returns:
            String: The actual if-else code as a string
        """
        code = ""
        tabs = "".join(['\t' for i in range(level)])


        if head.prediction is not None:
            return tabs + "    return " + str(int(np.argmax(head.prediction))) + ";\n" ;
        else:
                code += tabs + "    if(pX[" + str(head.feature) + "] <= " + str(head.split) + "){\n"
                code += "label_{number}:\n".replace("{number}", str(head.leftChild))                
                code += self.getImplementation(treeID, head.leftChild, level + 1)
                code += tabs + self.getProbChild(head, head.leftChild)
                # else
                code += tabs + "    } else {\n"
                code += "label_{number}:\n".replace("{number}", str(head.rightChild))                
                code += self.getImplementation(treeID, head.rightChild, level + 1)
                code += tabs + self.getProbChild(head, head.rightChild)
                code += tabs + "    }\n"

        return code

    def getCode(self, tree, treeID, numClasses):
        """ Generate the actual if-else implementation for a given tree

        Args:
            tree (TYPE): The tree
            treeID (TYPE): The id of this tree (in case we are dealing with a forest)

        Returns:
            Tuple: A tuple (headerCode, cppCode), where headerCode contains the code (=string) for
            a *.h file and cppCode contains the code (=string) for a *.cpp file
        """
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
                         return """     __builtin_prefetch ( &&label_{number} );\n""".replace("{number}", str(node.rightChild))
                    else:
                         return """     __builtin_prefetch ( &&label_{number} );\n""".replace("{number}", str(node.leftChild))
                else:
                     return """     __builtin_prefetch ( &&label_{number} );\n""".replace("{number}", str(node.leftChild))
        else:
                if node.probRight is not None:
                     return """     __builtin_prefetch ( &&label_{number} );\n""".replace("{number}", str(node.rightChild))
                else:
                     return"""     __builtin_prefetch ( &&label_{number} );\n""".replace("{number}", str(head.id))


