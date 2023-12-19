"""
In this file the sectionalizing and tie switch details are specified. 
The DER details are also specified (in particular the ones with and without grid forming capability). 
Also the path for the DSS file containing the circuit information is specified.
The final DSS circuit which will be used by the environment is created.

---G_init is used to maintain order of nodes, edges and also the varying graph scenario
---G_base is used to simulate fault isolation
"""

import os
import networkx as nx
from  Environments.DSSdirect_13bus_loadandswitching.DSS_CircuitSetup import*

#------------------- User defined inputs to modify the standard test network ---------------------------------

sectional_swt=[{'no':1,'line':'632670'},
               {'no':2,'line':'671692'}]

tie_swt=[{'no':1,'from node':'646','from conn':'.2.3', 'to node':'684','to conn':'.1.3', 'length':2000,'code':'603'},
          {'no':2,'from node':'633','from conn':'.1.2.3','to node':'692','to conn':'.1.2.3','length':2000,'code':'601'}]

# Generic generators
#Jacobs, Nicholas, Shamina Hossain-McKenzie, and Adam Summers. "Modeling data flows with network calculus in cyber-physical systems: enabling feature analysis for anomaly detection applications." Information 12.6 (2021): 255.
generators=[{'no':1, 'bus':'645', 'numphase':1, 'phaseconn':'.2', 'size':20, 'kV':2.4, 'Gridforming':'No'},
            {'no':2, 'bus':'645', 'numphase':1, 'phaseconn':'.3', 'size':20, 'kV':2.4, 'Gridforming':'No'},
            {'no':3, 'bus':'634', 'numphase':3, 'phaseconn':'.1.2.3', 'size':1000, 'kV':0.48, 'Gridforming':'Yes'},
            {'no':4, 'bus':'684', 'numphase':1, 'phaseconn':'.1', 'size':50, 'kV':2.4, 'Gridforming':'No'},
            {'no':5, 'bus':'684', 'numphase':1, 'phaseconn':'.3', 'size':50, 'kV':2.4, 'Gridforming':'No'},
            {'no':6, 'bus':'680', 'numphase':3, 'phaseconn':'.1.2.3', 'size':1000, 'kV':4.16, 'Gridforming':'Yes'},
            {'no':7, 'bus':'675', 'numphase':3, 'phaseconn':'.1.2.3', 'size':500, 'kV':4.16,'Gridforming':'No'}]

substatn_id='sourcebus'

dispatch_loads = ['Load.671','Load.634a','Load.634b','Load.634c','Load.611','Load.652','Load.670b']

n_actions = len(sectional_swt) + len(tie_swt) + len(dispatch_loads) # the switching actions and load shedding/pickup
#------------ Define the network with additions of DER, BESS and switches -------------------------------------
def initialize():       
    FolderName=os.path.dirname(os.path.realpath(__file__))
    DSSfile=r""+ FolderName+ "/IEEE13Nodeckt.dss"
    # DSSfile="IEEE13Nodeckt.dss"
    DSSCktobj = CktModSetup(DSSfile,sectional_swt,tie_swt,generators) # initially the sectionalizing switches close and tie switches open
    DSSCktobj.dss.Solution.Solve() #solving snapshot power flows
    if DSSCktobj.dss.Solution.Converged():
       conv_flag=1
    else:
       conv_flag=0    
    G_init=graph_struct(DSSCktobj)
    return DSSCktobj,G_init,conv_flag

DSSCktobj,G_init,conv_flag= initialize() 

# G_init has both sectionalizing and tie switches 

#--------- Graph with normal operating topology (with only sectionalizing switches)--for outage simulation (fault isolation)
tie_edges=[]
i=DSSCktobj.dss.SwtControls.First()
while i>0:
    name=DSSCktobj.dss.SwtControls.Name()
    if name[:5]=='swtie':
       line=DSSCktobj.dss.SwtControls.SwitchedObj()
       br_obj=Branch(DSSCktobj,line)
       from_bus=br_obj.bus_fr.split('.')[0]
       to_bus=br_obj.bus_to.split('.')[0]
       tie_edges.append((from_bus,to_bus))
    i=DSSCktobj.dss.SwtControls.Next()
G_base = G_init.copy()    
G_base.remove_edges_from(tie_edges)

#---------------------Create a dictionary with name of generator element and corresponding buses and also one with blackstart capability---------------------------

Generator_Buses={} # list of dictionary create for opendss element extraction
Generator_BlackStart={}
i= DSSCktobj.dss.Generators.First()
while i>0:      
      elemName = f'Generator.{DSSCktobj.dss.Generators.Name()}' # full element name with device type included
      DSSCktobj.dss.Circuit.SetActiveElement(elemName)
      bus_connectn = DSSCktobj.dss.CktElement.BusNames()[0].split('.')[0]
      # phases = DSSCktobj.dss.CktElement.NumPhases()
      Generator_Buses[elemName]=bus_connectn
      num=int(elemName[-1])-1
      if generators[num]['Gridforming'] == 'Yes':
          Generator_BlackStart[elemName]=1
      else:
          Generator_BlackStart[elemName]=0 
      i = DSSCktobj.dss.Generators.Next()

#----------------- Create a dictionary with the name of loads and corresponding bus ---------------------
Load_Buses={}
i= DSSCktobj.dss.Loads.First()
while i>0:
      elemName = f'Load.{DSSCktobj.dss.Loads.Name()}'
      DSSCktobj.dss.Circuit.SetActiveElement(elemName)
      bus_connectn = DSSCktobj.dss.CktElement.BusNames()[0].split('.')[0]
      # phases = DSSCktobj.dss.CktElement.NumPhases()
      Load_Buses[elemName]=bus_connectn
      i = DSSCktobj.dss.Loads.Next()
      
#------------------ List of the network buses and bus connections -------------------------------------      
node_list=list(G_init.nodes())
edge_list=list(G_init.edges()) #the fixed set of edges
nodes_conn=[]
for bus in node_list:
    nodes_conn.append(Bus(DSSCktobj,bus).nodes)
    

#------ Assigning a black start indicator for generators--------------------------------
gen_buses = np.array(list(Generator_Buses.values())) # all the generator buses
gen_elems=  list(Generator_Buses.keys()) # all the generator elements
Gen_Info ={}
for n in node_list:
    blackstart_flag=0
    gen_names=[gen_elems[x]  for x in  np.where(gen_buses == n)[0]]
    if (len(gen_names)!=0):
        for g in gen_names:
            blackstart_flag = blackstart_flag + Generator_BlackStart[g] 
        Gen_Info[n]= {'Generators':gen_names, 'Blackstart':blackstart_flag}
    

#-----------End of simulation , now some code snippet for checking switches, voltages, currents
# Check switches status

AllSwitches=[]
i=DSSCktobj.dss.SwtControls.First()
while i>0:
    name=DSSCktobj.dss.SwtControls.Name()
    line=DSSCktobj.dss.SwtControls.SwitchedObj()
    br_obj=Branch(DSSCktobj,line)
    from_bus=br_obj.bus_fr
    to_bus=br_obj.bus_to
    # DSSCktobj.dss.Circuit.SetActiveElement(line)
    # if(DSSCktobj.dssCircuit.ActiveCktElement.IsOpen(1,0)):
    #     sw_status=0
    # else:
    #     sw_status=1
    sw_status=DSSCktobj.dss.SwtControls.Action()-1 
    AllSwitches.append({'switch name':name,'edge name':line, 'from bus':from_bus.split('.')[0], 'to bus':to_bus.split('.')[0], 'status':sw_status})
    i=DSSCktobj.dss.SwtControls.Next()    

SwitchLines=[(s['from bus'],s['to bus']) for s in AllSwitches]

    
# # Check Node Voltages
V_nodes=[]
for n in list(G_init.nodes()):
    V=Bus(DSSCktobj, n).Vmag
    temp_conn=Bus(DSSCktobj, n).nodes
    V_nodes.append({'name':n, 'Connection': temp_conn, 'Voltage':V})

# Check Line Currents
I_nodes=[]
for e in list(G_init.edges(data=True)):
    branchname=e[2]['label'][0]
    DSSCktobj.dss.Circuit.SetActiveElement(branchname)
    I=DSSCktobj.dss.CktElement.Powers()
    # I=Branch(DSSCktobj, branchname).Cap
    I_nodes.append({'name':branchname, 'Current':I})         
