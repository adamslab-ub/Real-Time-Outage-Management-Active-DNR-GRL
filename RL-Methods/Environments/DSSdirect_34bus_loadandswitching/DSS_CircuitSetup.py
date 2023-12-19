"""
In this file the DSS engine is set up.
The objects for circuit, bus, and branch are also set up for further use in the RL environment.
This file also includes function to modify the base DSS circuit with sectionalizing, tie switch, and generator information.
The translation of DSS Circuit into a graph structure is also defined here.
"""

import opendssdirect as dss 
import numpy as np
import math
import networkx as nx


class DSS():  # to initialize the DSS circuit object and extract results      
    def __init__(self,filename):         
        self.filename = filename
        self.dss=dss  
            
    def compile_ckt_dss(self): # Compiling the OpenDSS circuit
        self.dss.Basic.ClearAll()
        self.dss.Text.Command("compile [" + self.filename +"]")
        
        
# Bus class contains the bus object details
class Bus:
    def __init__(self, DSSCktobj, bus_name):
        """
        Inputs:
            circuit object
            bus name
        Contains:
            Vmag-  pu voltage magnitude at bus nodes (3 phase)
            Vang-  pu voltage angle at bus nodes (3 phase)
            nodes- node connection at bus
        """ 
        Vmag=np.zeros(3)
        #Vang=np.zeros(3)
        DSSCktobj.dss.Circuit.SetActiveBus(bus_name)
        V=DSSCktobj.dss.Bus.puVmagAngle() # pair of magnitude and angle of voltages in pu
        nodes=np.array(DSSCktobj.dss.Bus.Nodes()) #Node connection
        for indx in range(len(nodes)):
            Vmag[nodes[indx]-1]=V[int(indx*2)] #assigning the voltages acc to node connection
            #Vang[nodes[indx]-1]=V[int(indx*2)+1]
        # Vmin=min(v for v in Vmag if v > 0)
        # Vmax=max(Vmag)
        self.Vmag = Vmag # 3 phase pu voltage at the buses
        # self.Vang = Vang
        self.nodes=nodes
        # self.Vmax=Vmax
        # self.Vmin=Vmin
        

class Branch:  # to extract properties of branch
    def __init__(self, DSSCktobj, branch_fullname):
        """
        Inputs:
            circuit object
            branch name
        Contains:                
            bus_fr - from bus name
            bus_to - to bus name         
            nphases - number of phases
            Cap - average current flow
            
        """        
        # Calculating base current
        DSSCktobj.dss.Transformers.First()
        KVA_base=DSSCktobj.dss.Transformers.kVA() # S base
        KV_base=DSSCktobj.dss.Transformers.kV() #L-L V base
        I_base=KVA_base/(math.sqrt(3)*KV_base)
               
        DSSCktobj.dss.Circuit.SetActiveElement(branch_fullname)
        
        bus_connections=DSSCktobj.dss.CktElement.BusNames()
        bus1= bus_connections[0]
        bus2= bus_connections[1]   
        
        i=np.array(DSSCktobj.dss.CktElement.CurrentsMagAng())
        ctidx = 2 * np.array(range(0, min(int(i.size/ 4), 3)))
        I_mag = i[ctidx] #branch current in A
        #I_ang=i[ctidx + 1] #angle in deg
        #nphases=DSSCktobj.dss.CktElement.NumPhases()
        #MaxCap=DSSCktobj.dss.CktElement.EmergAmps()
        # MaxCap=DSSCktobj.dss.CktElement.NormalAmps()
       # https://sourceforge.net/p/electricdss/discussion/861976/thread/8aa13830/
       # Problem is that Line.650632 already exceeds normal amps in Opendss=400 A and 
       # Normal Amps in Kerstings book =530 A. So I will consider EmergAmps=600 A
        I_avg=(np.sum(I_mag))/I_base #average of all three phases in pu

        self.bus_fr=bus1
        self.bus_to=bus2
        #self.nphases=nphases
        self.Cap=I_avg
        # self.MaxCap=MaxCap 
        
        
def CktModSetup(DSSfile,sectional_swt,tie_swt,generators): # give tie switches and sectionalizing switches as input
    DSSCktobj= DSS(DSSfile) #create a circuit object
    DSSCktobj.compile_ckt_dss() #compiling the circuit #compiling should only be done once in the beginning
    #Setting the iteration limits higher and disabling the warning message window
    DSSCktobj.dss.Text.Command("Set Maxiterations=500")
    DSSCktobj.dss.Text.Command("Set maxcontroliter=5000")
    # Donot put control mode= OFF at all...it will not allow switch control
    DSSCktobj.dss.Basic.AllowForms(0) 

    #### Make switch additions #####
    
    for sline in sectional_swt:  # the sectionalizing switch control is established (the normal state is closed)
        DSSCktobj.dss.Text.Command(f"New swtcontrol.swSec{str(sline['no'])} SwitchedObj=Line.{sline['line']} Normal=c SwitchedTerm=1 Action=c") #normally close      
    
    for tline in tie_swt: # First create new lines corresponding to tie lines
        DSSCktobj.dss.Text.Command(f"New Line.{tline['from node']}{tline['to node']} Bus1={tline['from node']}{tline['from conn']} Bus2={tline['to node']}{tline['to conn']} LineCode={tline['code']} Length={str(tline['length'])} units=kft")
        DSSCktobj.dss.Text.Command(f"New swtcontrol.swTie{str(tline['no'])} SwitchedObj=Line.{tline['from node']}{tline['to node']} Normal=o SwitchedTerm=1 Action=o") #normally open  
        # Swobj='Line.'+ tline['from node'] + tline['to node']
        # DSSCktobj.dss.Circuit.SetActiveElement(Swobj)
        # DSSCktobj.dss.CktElement.Open(1,0)
        # DSSCktobj.dss.Text.Command('open ' + Swobj +' term=1')       #switching the line open
    # For switches if Normal State= 1 it is open and if Normal State= 2  it is close in DSS
    
    
    ### Make DER additions #####  
    
    ##---------------- Generator Object ----------------------------------------
    for gen in generators:
        # connection - default is wye (Y/LN)
        # https://sourceforge.net/p/electricdss/discussion/861976/thread/f81830f0/
        DSSCktobj.dss.Text.Command(f"New Generator.G{str(gen['no'])} bus1={gen['bus']}{gen['phaseconn']} Phases={str(gen['numphase'])} Kv={str(gen['kV'])} Kw={str(gen['size'])} Pf=0.85 Model=1")        
    

    ##---------------- Time series simulation (Load shapes)-------------------------------- 
    # DSSCktobj.dss.Text.Command("redirect Loadshapes.dss") # load multiplication factor
    # # PV profile already added when defining PV
    # DSSCktobj.dss.Text.Command("BatchEdit Load..* Daily=loadshape_multload") # To add the same loadshape to all loads
    
    return DSSCktobj


# ---------Graph formation and Adjacency matrix--------------#
def graph_struct(DSSCktobj):           
    G_original=nx.Graph()
    i=DSSCktobj.dss.PDElements.First() #Getting all power delivery elements
    while i>0:
          label_edge=[]
          e=DSSCktobj.dss.PDElements.Name()
          if ((e.split('.')[0]=='Line') or (e.split('.')[0]=='Transformer')): #only if its a line or transformer used in graph(capacitors avoided)
             branch_obj=Branch(DSSCktobj,e) #creating  a branch object instance with full name
             sr_node=branch_obj.bus_fr.split('.')[0] #extracting source bus of branch
             tar_node=branch_obj.bus_to.split('.')[0] #extracting target bus of branch
             name=e
             if (G_original.has_edge(sr_node,tar_node)):#if it already has an edge just add the new element name to existing label.
                 label_edge=[x  for x in G_original.edges[sr_node,tar_node]['label']]
                 label_edge.append(name)                 
                 G_original.edges[sr_node,tar_node]['label']=label_edge
                 
             elif (G_original.has_edge(tar_node,sr_node)):
                   label_edge=[x for x in G_original.edges[tar_node,sr_node]['label']]
                   label_edge.append(name)  
                   G_original.edges[tar_node,sr_node]['label']= label_edge
                   
             else:
                  label_edge.append(name)
                  G_original.add_edge(sr_node, tar_node, label= label_edge)
                  
          i=DSSCktobj.dss.PDElements.Next()
    
    return G_original
        