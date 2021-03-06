from ForestConverter import TreeConverter
import numpy as np
import heapq

class RootNativeTreeConverter(TreeConverter): # like a super class
    def __init__(self, dim, namespace, featureType):
        super().__init__(dim, namespace, featureType)

    def getArrayLenType(self, arrLen):
            arrayLenBit = int(np.log2(arrLen)) + 1
            if arrayLenBit <= 8:
                    arrayLenDataType = "unsigned char"
            elif arrayLenBit <= 16:
                    arrayLenDataType = "unsigned short"
            else:
                    arrayLenDataType = "unsigned int"
            return arrayLenDataType

        # 'Abstract method'
    def getImplementation(self, head, treeID):
        raise NotImplementedError("This function should not be called directly, but only by a sub-class")

        # set header code
    def getHeader(self, splitType, treeID, arrLen, numClasses):
            dimBit = int(np.log2(self.dim)) + 1 if self.dim != 0 else 1

            if dimBit <= 8:
                    dimDataType = "unsigned char"
            elif dimBit <= 16:
                    dimDataType = "unsigned short"
            else:
                    dimDataType = "unsigned int"

            featureType = self.getFeatureType()
            if (numClasses == 2):
                headerCode = """struct {namespace}_Node{treeID} {
                        //bool isLeaf;
                        //unsigned int prediction;
                        {dimDataType} feature;
                        {splitType} split;
                        {arrayLenDataType} leftChild;
                        {arrayLenDataType} rightChild;
                        unsigned char indicator;
                        {arrayLenDataType} probChild;

                };\n""".replace("{namespace}", self.namespace) \
                           .replace("{treeID}", str(treeID)) \
                           .replace("{splitType}",splitType) \
                           .replace("{dimDataType}",dimDataType) \
                           .replace("{arrayLenDataType}", self.getArrayLenType(arrLen))
            else: # here is no prediction included
                headerCode = """struct {namespace}_Node{treeID} {
                           //bool isLeaf;
                            {dimDataType} feature;
                            {splitType} split;
                            {arrayLenDataType} leftChild;
                            {arrayLenDataType} rightChild;
                            unsigned char indicator;
                            {arrayLenDataType} probChild;
                };\n""".replace("{namespace}", self.namespace) \
                       .replace("{treeID}", str(treeID)) \
                       .replace("{splitType}",splitType) \
                       .replace("{dimDataType}",dimDataType) \
                       .replace("{arrayLenDataType}", self.getArrayLenType(arrLen))

            headerCode += "inline unsigned int {namespace}_predict{treeID}({feature_t} const pX[{dim}]);\n" \
                                            .replace("{treeID}", str(treeID)) \
                                            .replace("{dim}", str(self.dim)) \
                                            .replace("{namespace}", self.namespace) \
                                            .replace("{feature_t}", featureType)
            return headerCode

    def getCode(self, tree, treeID, numClasses):
 
            tree.getProbAllPaths()
            cppCode, arrLen = self.getImplementation(tree.head, treeID)

            if self.containsFloat(tree):
                splitDataType = "float"
            else:
                lower, upper = self.getSplitRange(tree)

                bitUsed = 0
                if lower > 0:
                    prefix = "unsigned"
                    maxVal = upper
                else:
                    prefix = ""
                    bitUsed = 1
                    maxVal = max(-lower, upper)

                splitBit = int(np.log2(maxVal) + 1 if maxVal != 0 else 1)

                if splitBit <= (8-bitUsed):
                    splitDataType = prefix + " char"
                elif splitBit <= (16-bitUsed):
                    splitDataType = prefix + " short"
                else:
                    splitDataType = prefix + " int"
            headerCode = self.getHeader(splitDataType, treeID, arrLen, numClasses)

            return headerCode, cppCode

class RootNativeTreeConverter(RootNativeTreeConverter):
    def __init__(self, dim, namespace, featureType):
            super().__init__(dim, namespace, featureType)

    def getHeader(self, splitType, treeID, arrLen, numClasses): #getHeader without numClass usage
            dimBit = int(np.log2(self.dim)) + 1 if self.dim != 0 else 1

            if dimBit <= 8:
                    dimDataType = "unsigned char"
            elif dimBit <= 16:
                    dimDataType = "unsigned short"
            else:
                    dimDataType = "unsigned int"

            featureType = self.getFeatureType()
            headerCode = """struct {namespace}_Node{treeID} {
                    bool isLeaf;
                    unsigned int prediction;
                    {dimDataType} feature;
                    {splitType} split;
                    {arrayLenDataType} leftChild;
                    {arrayLenDataType} rightChild;
                    {arrayLenDataType} probChild;
            };\n""".replace("{namespace}", self.namespace) \
                       .replace("{treeID}", str(treeID)) \
                       .replace("{arrayLenDataType}", self.getArrayLenType(arrLen)) \
                       .replace("{splitType}",splitType) \
                       .replace("{dimDataType}",dimDataType)

            headerCode += "inline unsigned int {namespace}_predict{treeID}({feature_t} const pX[{dim}]);\n" \
                                            .replace("{treeID}", str(treeID)) \
                                            .replace("{dim}", str(self.dim)) \
                                            .replace("{namespace}", self.namespace) \
                                            .replace("{feature_t}", featureType)

            return headerCode

    def getImplementation(self, head, treeID):
            arrayStructs = []
            nextIndexInArray = 1 # always one index ahead of current index

            # Breitensuche
            nodes = [head] # nodes is an array initiated with the root of the tree
            while len(nodes) > 0:
                node = nodes.pop(0) # return first element which is a tree
                entry = [] # temporary array

                if node.prediction is not None:
                    entry.append(1) #isLeaf
                    entry.append(int(np.argmax(node.prediction))) # prediction
                    entry.append(0) # feature
                    entry.append(0) # split
                    entry.append(0) # leftChild
                    entry.append(0) # rightChild
                else:
                    entry.append(0) # isLeaf
                    entry.append(0) # Constant prediction
                    entry.append(node.feature) # feature
                    entry.append(node.split) # split
                    entry.append(nextIndexInArray) # leftChild
                    nextIndexInArray += 1 # +1 for rightChild
                    entry.append(nextIndexInArray) # rightChild
                    nextIndexInArray += 1 # +1 for next index
                # nodes is empty now, append with BFS algo
                    nodes.append(node.leftChild) # fill with leftChild at the end of the array
                    nodes.append(node.rightChild) # fill with rightChild at the end of the array

                    entry.append(head) # prefetch root

                arrayStructs.append(entry) # temporary array 'entry' will be append on further used arrayStructs
        
            featureType = self.getFeatureType()
            arrLen = len(arrayStructs)
            
            cppCode = "#include <iostream>\n"

            cppCode += "{namespace}_Node{treeID} const tree{treeID}[{N}] = {" \
                    .replace("{treeID}", str(treeID)) \
                    .replace("{N}", str(len(arrayStructs))) \
                    .replace("{namespace}", self.namespace)

            for e in arrayStructs: # format for C++
                    cppCode += "{"
                    for val in e:
                            cppCode += str(val) + ","
                    cppCode = cppCode[:-1] + "},"
            cppCode = cppCode[:-1] + "};"

            cppCode += """
                    inline unsigned int {namespace}_predict{treeID}({feature_t} const pX[{dim}]){
                            {arrayLenDataType} i = 0;
                            while(!tree{treeID}[i].isLeaf) {
                                    if (pX[tree{treeID}[i].feature] <= tree{treeID}[i].split){
                                        i = tree{treeID}[i].leftChild;
                                    } else {
                                        i = tree{treeID}[i].rightChild;
                                    }
                                    __builtin_prefetch ( &tree{treeID}[tree{treeID}[i].probChild] );
                            }
                            return tree{treeID}[i].prediction;
                    }
            """.replace("{treeID}", str(treeID)) \
               .replace("{dim}", str(self.dim)) \
               .replace("{namespace}", self.namespace) \
               .replace("{arrayLenDataType}",self.getArrayLenType(len(arrayStructs))) \
               .replace("{feature_t}", featureType)

            return cppCode, arrLen