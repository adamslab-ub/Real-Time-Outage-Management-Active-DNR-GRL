"""
In this file the functions to evaluate the state, reward are defined and also the action is implemented
"""
import numpy as np
from Environments.DSSdirect_34bus_loadandswitching.DSS_Initialize import*

# To get the details of switches and their status
def switchInfo(DSSCktobj):
    #Input:   DSS Circuit Object
    #Returns: A list of dictionaries which contains: 
    #         The name of the switch.
    #         The associated line of the switch (the edge label).
    #         The status of the switch in the DSS Circuit object
    
    AllSwitches = []
    i = DSSCktobj.dss.SwtControls.First()
    while i>0:
        name = DSSCktobj.dss.SwtControls.Name()
        line = DSSCktobj.dss.SwtControls.SwitchedObj()
        DSSCktobj.dss.Circuit.SetActiveElement(line)
        sw_status = DSSCktobj.dss.SwtControls.Action()-1
        AllSwitches.append({'switch name':name,'edge name':line, 'status':sw_status})        
        i=DSSCktobj.dss.SwtControls.Next()

    return AllSwitches     


def get_state(DSSCktobj, G, edgesout):
    #Input: DSS Circuit Object, the equivalent Graph representation, and the out edges
    #Returns: Dictionary to indicate state which includes:
              # Total Energy Supplied, bus voltage (nodes), branch powerflow (edges), adjacency, voltage and convergence violations
    
        
    Adj_mat = nx.adjacency_matrix(G,nodelist=node_list) # adjacency matrix keeping node order fixed

    # Estimating the total energy supplied to the end users given the state encompassed in DSS Circuit
    DSSCktobj.dss.Transformers.First()
    KVA_base=DSSCktobj.dss.Transformers.kVA() #To convert into per unit

    En_Supply=0
    Total_Demand=0
    for ld in list(DSSCktobj.dss.Loads.AllNames()): # For each load
        DSSCktobj.dss.Circuit.SetActiveElement(f"Load.{ld}") #set the load as the active element
        S=np.array(DSSCktobj.dss.CktElement.Powers()) # Power vector (3 phase P and Q for each load)
        ctidx = 2 * np.array(range(0, min(int(S.size/ 2), 3)))
        P = S[ctidx] #active power in KW
        Q = S[ctidx + 1] #reactive power in KVar
        if (np.isnan(P).any()) or (np.isnan(Q).any()):
            Power_Supp = 0    # Nodes which are isolated with loads but no generators return nan-- ignore that(consider as inactive)  
        else:
            Power_Supp = sum(P) # total active power supplied at load
        if math.isnan(Power_Supp):
            Power_Supp = 0
        Demand = float (DSSCktobj.dss.Properties.Value('kW'))  
        En_Supply = En_Supply + Power_Supp
        Total_Demand =  Total_Demand + Demand  
   
    if Total_Demand !=0:
        En_Supply_perc = En_Supply/Total_Demand
    else:
        En_Supply_perc = -1   
 
         
    # Extracting the pu node voltages at all buses    
    Vmagpu=[]
    active_conn=[]
    for b in node_list:    
        V = Bus(DSSCktobj,b).Vmag
        active_conn.append(Bus(DSSCktobj,b).nodes)
        temp_flag = np.isnan(V) # Nodes which are isolated with loads but no generators return nan-- ignore that(consider as inactive)
        if np.any(temp_flag,where=True):
            V[temp_flag] = 0
            temp_conn=[n for n in active_conn[node_list.index(b)] if temp_flag[n-1]==False] #the bus nodes active are only those with numerical(not nan)
            active_conn[node_list.index(b)]=np.array(temp_conn) #only active node connections
        Vmagpu.append(V)
    
    # Extracting the pu average branch currents(also includes the open branches)           
    
    I_flow=[]
    for e in G_init.edges(data=True):
        branchname=e[2]['label'][0]
        I=Branch(DSSCktobj, branchname).Cap
        I_flow.append(I)

    # The convergence test and violation penalty   
    if DSSCktobj.dss.Solution.Converged():
        conv_flag=1
        Conv_const=0
    else:
        conv_flag=0
        Conv_const=10# NonConvergence penalty   

    
    # The voltage violation
    V_viol=Volt_Constr(Vmagpu,active_conn)
     
    # To mask those switches which are out (including line and load switches)
    SwitchMasks=[]
    for x in SwitchLines:
        if x in edgesout:
            SwitchMasks.append(1)
        else:
            SwitchMasks.append(0)
    for y in dispatch_loads:
        SwitchMasks.append(0)
    
    
    return {
        "EnergySupp":np.array([En_Supply_perc]),
        "NodeFeat(BusVoltage)":np.array(Vmagpu), 
        "EdgeFeat(Branchflow)":np.array(I_flow),
        "Adjacency":np.array(Adj_mat.todense()), 
        "VoltageViolation":np.array([V_viol]), 
        "ConvergenceViolation":np.array([Conv_const]),
        "ActionMasking":np.array(SwitchMasks)}


def take_action(action, out_edges):
    #Input :object of type DSSObj.ActiveCircuit (dss interface for OpenDSS Circuit)
    #Input: action multi binary type. i.e., the status of each switch if it is 0 open and 1 close,
    #  the status of loads appended in action after switches: 0 shed, 1:picked up
    #Returns:the circuit object with action implemented (and slack assigned), also the graph scenario

    DSSCktObj,G_init,conv_flag = initialize() # local DSS object for just implementing action
    G_sc = G_init.copy() # Copy to create graph scenario

   # -------------Implement Line Switching Action on DSSCircuit Object
    switch_actionidx = 0
    i=DSSCktObj.dss.SwtControls.First()
    while (i>0):
        switch_actionidx= i-1
        if action[switch_actionidx]==0: # if 0- open line
            # DSSCktobj.dssCircuit.ActiveCktElement.Open(1,0)
            DSSCktObj.dss.Text.Command('Swtcontrol.' + DSSCktObj.dss.SwtControls.Name() + '.Action=o')
            # DSSCktobj.dssText.command='open ' + Swobj +' term=1'       #switching the line open
        else:
            # DSSCktobj.dssCircuit.ActiveCktElement.Close(1,0)
            DSSCktObj.dss.Text.Command('Swtcontrol.'+ DSSCktObj.dss.SwtControls.Name() + '.Action=c')
            # DSSCktobj.dssText.command='close ' + Swobj +' term=1'      #switching the line close
        i=DSSCktObj.dss.SwtControls.Next()

    DSSCktObj.dss.Solution.Solve()

   # -------------Implement Load Shedding Action on DSSCircuit Object

    for load_actionidx in range(switch_actionidx + 1, n_actions):
        loadname = dispatch_loads[load_actionidx-switch_actionidx-1]
        if action[load_actionidx] == 0: # If load is shed- disable the load
           DSSCktObj.dss.Circuit.SetActiveElement(loadname)
           DSSCktObj.dss.Text.Command(loadname + '.enabled="False"')
        DSSCktObj.dss.Solution.Solve()

    #----Disable outage lines from DSSCircuit and also from Graph Scenario
    for o_e in out_edges:
        (u,v) = o_e        
        if G_sc.has_edge(u,v):
           G_sc.remove_edge(u,v) # Remove the edge in graph domain
        # Remove the element from the DSSCktobj
        branch_name = G_init.edges[o_e]['label'][0]
        DSSCktObj.dss.Circuit.SetActiveElement(branch_name)
        DSSCktObj.dss.Text.Command(f'Open {branch_name} term=1')
        DSSCktObj.dss.Solution.Solve() 


    #---------- Also remove the open switches from Graph Scenario
    i=DSSCktObj.dss.SwtControls.First()
    while i>0:
          line=DSSCktObj.dss.SwtControls.SwitchedObj()
          if DSSCktObj.dss.SwtControls.Action() == 1: #Open is 1 in DSS
             b_obj=Branch(DSSCktObj, line)
             u=b_obj.bus_fr.split('.')[0]
             v=b_obj.bus_to.split('.')[0]
             if G_sc.has_edge(u,v):
                G_sc.remove_edge(u,v) # Remove the edge in graph domain
          i=DSSCktObj.dss.SwtControls.Next()


    # #----- Finding network components and find virtual slack ------#
    Components= list(nx.connected_components(G_sc)) #components formed due to outage
    Virtual_Slack=[] # for each component not connected to sourcebus...we will assign a slack
    if len(Components) >1 : #Only if there exists a network component unconnected to sourcebus virtual slack is assigned
        for C in Components: #for each component
            if substatn_id not in C: # for the component unconnected to sourcebus
                Slack_DER={'name':'','kVA':0}
                # Find the DER corresponding to slack bus (largest grid forming DER) in component
                for gen_bus, gen_info in Gen_Info.items():
                    if gen_bus in C and gen_info['Blackstart']==1: #if generator is present and has gridforming capability
                       kva_val=0
                       for gen_name in gen_info['Generators']: #get total KVA at node
                           DSSCktObj.dss.Circuit.SetActiveElement(gen_name)
                           kva_val= kva_val + float(DSSCktobj.dss.Properties.Value('kVA'))
                       if kva_val > Slack_DER ['kVA'] : # if multiple grid forming DERs, largest grid forming DER is slack
                          Slack_DER['kVA'] = kva_val
                          Slack_DER['name']= ('bus_'+ gen_bus)
                Virtual_Slack.append(Slack_DER)

    #---- Assign slack bus in DSSCkt at the buses with virtual slack in different graph components
    for vs in Virtual_Slack:
        Vs_name=vs['name']
        if Vs_name != '':
            Vs_locatn=Vs_name.split('_')[1]
            Vs_MVA = Vs_MVAsc3 = vs['kVA']/1000 #MVA and MVAsc3 are set to be same
            Vs_MVAsc1 = Vs_MVAsc3/3 # MVAsc1 approax 1/3 of MVAsc3

            DSSCktObj.dss.Circuit.SetActiveBus(Vs_locatn)
            # DSSCktobj.dssBus.kVBase gives the per phase (phase to neutral) voltage
            Vs_kv =DSSCktObj.dss.Bus.kVBase() * math.sqrt(3) # this has to be phase to phase
            DSSCktObj.dss.Text.Command(f"New Vsource.{Vs_name}  bus1={Vs_locatn}  basekV={str(Vs_kv)}  phases=3  Pu=1.00  angle=30  baseMVA={str(Vs_MVA)}  MVAsc3={str(Vs_MVAsc3)}  MVAsc1={str(Vs_MVAsc1)}  enabled=yes")
            # print(Vs_MVAsc3)
            # print(Vs_MVAsc1)
            # DSSCktobj.dssText.command = 'Formedit'+ ' Vsource.' +Vs_name
            for gens in Gen_Info[Vs_locatn]['Generators']:
                DSSCktObj.dss.Text.Command(gens + '.enabled=no')
    DSSCktObj.dss.Solution.Solve()

    return DSSCktObj,G_sc


# Constraint for voltage violation 
def Volt_Constr(Vmagpu,active_conn):
    #Input: The pu magnitude of node voltages at all buses, node activated or node phase of all buses
    Vmax=1.10
    Vmin=0.90    
    V_Viol= []
    
    for i in range(len(active_conn)):
        for phase_co in active_conn[i]:
            if (Vmagpu[i][phase_co-1]<Vmin):  
                viol = abs(Vmin-Vmagpu[i][phase_co-1])/Vmin
                V_Viol.append(viol)
            if (Vmagpu[i][phase_co-1]>Vmax): 
                viol = abs(Vmagpu[i][phase_co-1]-Vmax)/Vmax                   
                V_Viol.append(viol)
    if len(V_Viol)!=0:
        V_ViolSum = (np.sum(V_Viol))/(len(G_init.nodes())*3)
    else: 
        V_ViolSum = 0
            
    return V_ViolSum


def get_reward(observ_dict):
    #Input: A dictionary describing the state of the network
    # ----#Output: reward- minimize load outage, penalize non convergence and closing of outage lines

    if observ_dict['ConvergenceViolation'] > 0 or math.isinf(observ_dict['VoltageViolation']):
        reward = np.array([0.0])
    else:

        reward = observ_dict['EnergySupp'] - observ_dict['VoltageViolation']

    # reward= observ_dict['EnergySupp']-observ_dict['ConvergenceViolation']-observ_dict['VoltageViolation']
    # print(observ_dict['VoltageViolation'], observ_dict['EnergySupp'])
    return reward

