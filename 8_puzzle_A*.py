#lab9
total_cost=0
total_cost1=0
  
def possible_operation(matrix):
    import numpy as np
    matrix=np.array(matrix)
    pos_row=0
    pos_col=0
    row,col=matrix.shape
    status=0
    operation=[]
    for i in range(0,row):
      if(status==1):
        break
      for j in range(0,col):
        if(matrix[i][j]==-1):
          pos_row=i
          pos_col=j
          status=1
          break
    row=pos_row
    col=pos_col
    if(row==0):
      operation.append("bottom")
    if(row==1):
      operation.append("top")
      operation.append("bottom")
    if(row==2):
      operation.append("top")
    if(col==0):
      operation.append("right")
    if(col==1):
      operation.append("left")
      operation.append("right")
    if(col==2):
      operation.append("left")
    return operation,pos_row,pos_col

def perform_operation(matrix_original,operation,row,col):
  new_matrix=[]
  import numpy as np
  matrix_original=np.array(matrix_original)
  if("top" in operation):
    matrix=matrix_original.copy()
    v1=matrix[row][col]
    v2=matrix[row-1][col]
    matrix[row][col]=v2
    matrix[row-1][col]=v1
    new_matrix.append(matrix)
  if("bottom" in operation):
    matrix=matrix_original.copy()
    v1=matrix[row][col]
    v2=matrix[row+1][col]
    matrix[row][col]=v2
    matrix[row+1][col]=v1
    new_matrix.append(matrix)
  if("left" in operation):
    matrix=matrix_original.copy()
    v1=matrix[row][col]
    v2=matrix[row][col-1]
    matrix[row][col]=v2
    matrix[row][col-1]=v1
    new_matrix.append(matrix)
  if("right" in operation):
    matrix=matrix_original.copy()
    v1=matrix[row][col]
    v2=matrix[row][col+1]
    matrix[row][col]=v2
    matrix[row][col+1]=v1
    new_matrix.append(matrix)
  return new_matrix

def cost_calculation_minimum_matrix_found(matrix_list,goal,g_n):
  
  import numpy as np
  min_value=100
  index=0
  counter=-1
  global total_cost1
  
  for i in matrix_list:
    counter=counter+1
    k1=np.array(i)
    k2=np.array(goal)
    temp=k1-k2
    row,col=temp.shape
    cost_value=0
    for i1 in range(0,row):
      for j1 in range(0,col):
        if(temp[i1][j1]!=0):
          cost_value=cost_value+1
    
    total_cost1=total_cost1+cost_value+g_n
    if((cost_value+g_n)<min_value):
      min_value=cost_value+g_n
      index=counter
  print("cost",min_value,g_n,index)
  return matrix_list[index]

def cost_calculation_minimum_matrix_found_manhattan(matrix_list,goal,g_n):
  
  import numpy as np
  min_value=100
  index=0
  counter=-1
  global total_cost
  
  for i in matrix_list:
    d1=dict()
    d1['1']=[0,0]
    d1['2']=[0,0]
    d1['3']=[0,0]
    d1['4']=[0,0]
    d1['5']=[0,0]
    d1['6']=[0,0]
    d1['7']=[0,0]
    d1['8']=[0,0]
    d2=dict()
    d2['1']=0
    d2['2']=0
    d2['3']=0
    d2['4']=0
    d2['5']=0
    d2['6']=0
    d2['7']=0
    d2['8']=0
    counter=counter+1
    k1=np.array(i)
    k2=np.array(goal)

    for i1 in range(1,9):
      for i11 in range(0,3):
        for j11 in range(0,3):
          if(i1==k1[i11][j11]):
            d1[str(i1)]=[i11,j11]
      for i11 in range(0,3):
        for j11 in range(0,3):
            if(k2[i11][j11]==i1):
              
              d2[str(i1)]=np.sum(np.abs(np.array(d1[str(i1)])-np.array([i11,j11])))
    row,col=k1.shape
    cost_value=0
    for i1 in range(1,8):
        
          cost_value=cost_value+d2[str(i1)]
    
    total_cost=total_cost+cost_value+g_n
    if((cost_value+g_n)<min_value):
      min_value=cost_value+g_n
      index=counter
  
  return matrix_list[index]

def cost(source,destination):
  import numpy as np
  source=np.array(source)
  destination=np.array(destination)
  temp=source-destination
  row,col=temp.shape
  
  cost_value=0
  for i in range(0,row):
    for j in range(0,col):
      if(temp[i][j]!=0):
        cost_value=cost_value+1
  return cost_value

def PARTIAL_SEARCH_TREE(START,GOAL):
  import numpy as np
  graph=[]
  s1=np.array(START)
  pattern=[]
  nodes_visited=1
  while(True):
    pattern.append(s1)
    if(cost(s1,GOAL)==0):
      break

    operation,row,col=possible_operation(s1)
    nodes_visited=nodes_visited+len(operation)
    operation_matrix=perform_operation(s1,operation,row,col)
    for i in operation_matrix:
      temp=[]
      temp.append(s1)
      temp.append(np.array(i))
      graph.append(temp)
    new_matrix=cost_calculation_minimum_matrix_found(operation_matrix,GOAL,len(pattern))
    
    s1=new_matrix
  for i in pattern:
    print(i,"\n\n")
  print("total visited nodes",nodes_visited)
  return graph

def PARTIAL_SEARCH_TREE_manhattan(START,GOAL):
  import numpy as np
  graph=[]
  s1=np.array(START)
  pattern=[]
  nodes_visited=1
  while(True):
    pattern.append(s1)
    if(cost(s1,GOAL)==0):
      break
    
    operation,row,col=possible_operation(s1)
    nodes_visited=nodes_visited+len(operation)
    operation_matrix=perform_operation(s1,operation,row,col)
    for i in operation_matrix:
      temp=[]
      temp.append(s1)
      temp.append(np.array(i))
      graph.append(temp)
    new_matrix=cost_calculation_minimum_matrix_found_manhattan(operation_matrix,GOAL,len(pattern))
    
    s1=new_matrix
  for i in pattern:
    print(i,"\n\n")
  print("number of node visited",nodes_visited)
  return graph

START=[[-1,5,2],[1,8,3],[4,7,6]]
GOAL=[[1,2,3],[4,5,6],[7,8,-1]]

print("Partial Search Tree\n")
graph1=PARTIAL_SEARCH_TREE(START,GOAL)  #it perform misplace tiles as distance as heuristic function
print("total cost\n",total_cost1)
print("graph for misplaces tiles\n",graph1)
print("Partial Search Tree using manhattan distance\n") #it perform manhattan  distance and number of moves required as heuristic function
print("Partial Search Tree manhattan\n")
graph2=PARTIAL_SEARCH_TREE_manhattan(START,GOAL)
print("total cost\n",total_cost)
print("graph for manhattan distance and number of moves required tiles\n",graph2)
