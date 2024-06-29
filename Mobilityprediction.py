from unimol_tools import MolTrain, MolPredict
import pandas as pd 
import numpy as np
import warnings
from sklearn.metrics import mean_squared_error
warnings.filterwarnings("ignore")
import math
from rdkit import Chem
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_linnerud
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from rdkit.Chem import AllChem
import re
from rdkit.Chem import rdmolops
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.datasets import make_blobs  # 仅用于生成示例数据
import warnings
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
warnings.filterwarnings('ignore')
from sklearn.metrics import r2_score
from math import sqrt
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import scipy.stats
import shap
from rdkit.Chem import Draw

def Get_COS2(x1):#计算（N，19）形式数据的cos2
    PHI=[]
    PP=[]
    if len(x1[0])>1:
        x=x1
    if len(x1[0])==1:
        x=[]
        for i in range(int(len(x1)/19)):
            a=[]
            for j in range(19):
                a.append(x1[19*i+j])
            x.append(a)
    for i in range(len(x)):
        P=[]
        phi=0
        for j in range(19):
            P.append(math.exp(-x[i][j]/(1.987072*0.001*298)))
        A=1/sum(P)
        ang=[math.cos(math.pi*ii/18) for ii in range(19)]
        for ii in range(len(P)):
            P[ii]=P[ii]*A
            phi+=(ang[ii]**2)*P[ii]
        PP.append(P)
        PHI.append(phi)
    return PHI
    
def Get_id_bysymbol(combo,symbol):#获取symbol标记的元素序号
    for at in combo.GetAtoms():
        if at.GetSymbol()==symbol:
            return at.GetIdx()

def Get_neiid_bysymbol(combo,symbol):#获取symbol相邻的第一个的元素序号
    for at in combo.GetAtoms():
        if at.GetSymbol()==symbol:
            at_nei=at.GetNeighbors()[0]
            return at_nei.GetIdx()
        
def combine2frag(Amol,Fr,Bmol,Cs):#Fr和Cs表记位置连接
    combo = Chem.CombineMols(Amol,Bmol)
    Fr_NEI_ID=Get_neiid_bysymbol(combo,Fr)
    Cs_NEI_ID=Get_neiid_bysymbol(combo,Cs)
    edcombo = Chem.EditableMol(combo)
    edcombo.AddBond(Fr_NEI_ID,Cs_NEI_ID,order=Chem.rdchem.BondType.SINGLE)

    Fr_ID=Get_id_bysymbol(combo,Fr)
    edcombo.RemoveAtom(Fr_ID)
    back = edcombo.GetMol()


    Cs_ID=Get_id_bysymbol(back,Cs)

    edcombo=Chem.EditableMol(back)
    edcombo.RemoveAtom(Cs_ID)
    back = edcombo.GetMol()
    smi= Chem.MolToSmiles(back)
    return smi

def Get_U(U):#讲单元的smiles描述符转化为无连符号的smiles
    U1=U.replace('Fr','H')
    U2=U1.replace('Cs','H')
    m=Chem.MolToSmiles(Chem.RemoveHs(Chem.MolFromSmiles(U2)))
    return m

def Process_smi(data1):#data1为含有smiles的文本list，形式为N个['smi1','smi2','smi3'],转化为单独的可识别的smiles list
    smi=[]
    for i in range(len(data1)):
        #print(i,data1[i])
        dimer1=data1[i][1:len(data1[i])-1].split(',')
        a=[]
        for j in range(len(dimer1)):
            for z in range(len(dimer1[j])):
                if dimer1[j][z]=="'":
                    a.append(dimer1[j][z+1:len(dimer1[j])-1])
                    break
                else:
                    pass
        smi.append(a)
    return smi
    
def unique_molecules(smiles_list):
    unique_smiles = set()  # 用于存储唯一的SMILES字符串
    unique_mols = []  # 存储去重后的分子对象列表
    for smi in smiles_list:
        mol = Chem.RemoveHs(Chem.MolFromSmiles(smi))
        if mol is not None:
            canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
            #print(canonical_smiles)
            if canonical_smiles not in unique_smiles:
                #print(unique_smiles)
                unique_smiles.add(canonical_smiles)
                unique_mols.append(mol)
        else:
            print('Can not convert mol',smiles_list.index(smi),smi)
    return unique_mols,list(unique_smiles)
    
def SMI_Prediction(smi,smi_name,module):#smi为待预测分子的smiles，smi_name为保存的文件名称,module为加载的模型名称
    if not isinstance(smi, dict):
        data_smi=pd.DataFrame(index=[i for i in range(len(smi))],columns=['SMILES']+[f'TARGET_{i+1}' for i in range(19)])
        #print(len(smi))
        data_smi['SMILES']=smi
        #print(data_smi.shape[0])
        #data_smi['']
        for i in range(19):
            data_smi[f'TARGET_{i+1}']=0
        data_smi.to_csv(smi_name,index=None)
    
        predm = MolPredict(load_model=module)
        pred_y = predm.predict(smi_name)
        for i in range(len(pred_y)):
            if len(pred_y[0])>1:
                for j in range(len(pred_y[0])):
                    data_smi[f'TARGET_{j+1}'][i]=pred_y[i][j]
            else:
                data_smi['TARGET_0']=pred_y[i]
        data_smi.to_csv(smi_name,index=None)
    else:
        predm = MolPredict(load_model=module)
        pred_y = predm.predict(smi)
    return pred_y#,data_smi、
    
def Get_U_COS2(smi,U_SMI,pred_E):#smi:原本集合[[smi1-1,smi1-2],[smi2-1,smi2-2,smi2-3]],,U_SMIL所有的Smiles形式为1维数据，pred_E
    cos2=Get_COS2(pred_E)
    COS2=[]
    U_SMI=[Chem.MolToSmiles(Chem.MolFromSmiles(x)) for x in U_SMI]
    for i in range(len(smi)):
        a=[]
        for j in range(len(smi[i])):
            mol = Chem.MolFromSmiles(smi[i][j])
            if mol is not None:
                canonical_smiles = Chem.MolToSmiles(mol)
                idx=U_SMI.index(canonical_smiles)
                a.append(cos2[idx])
        COS2.append(a)
    return COS2

#侧链信息获取
def Get_path(mol,id1,id2):
    #获取原子路径
    path = rdmolops.GetShortestPath(mol, id1, id2)
    # 获取路径上的键
    bonds = [mol.GetBondBetweenAtoms(path[i], path[i+1]).GetIdx() for i in range(len(path)-1)]
    return path,bonds

def Get_Mol(mol,c):#原分子mol，获取的原子序号list：c
    #mol=Chem.AddHs(mol)
    atom_indices = c  # 定义一个包含要提取的原子的标号列表（无需按连接顺序）
    editable_mol = Chem.EditableMol(Chem.Mol())# 创建一个可编辑的分子对象
    old_to_new_index = {}# 创建一个字典，用于映射旧原子索引到新原子索引
    #print(mol.GetNumAtoms())
    for old_index in atom_indices:# 添加所需的原子到可编辑的分子中，并记录映射
        #print(old_index)
        atom = mol.GetAtomWithIdx(old_index)
        new_index = editable_mol.AddAtom(atom)
        old_to_new_index[old_index] = new_index
    for bond in mol.GetBonds():# 添加所需的键到可编辑的分子中
        if bond.GetBeginAtomIdx() in atom_indices and bond.GetEndAtomIdx() in atom_indices:
            begin_atom_index = old_to_new_index[bond.GetBeginAtomIdx()]
            end_atom_index = old_to_new_index[bond.GetEndAtomIdx()]
            bond_type = bond.GetBondType()
            editable_mol.AddBond(begin_atom_index, end_atom_index, bond_type)
    result_mol = editable_mol.GetMol()# 获取最终的可编辑分子
    result_smiles = Chem.MolToSmiles(result_mol)# 将分子转换为SMILES字符串以进行可视化或其他操作
    
    return result_mol
    
def Get_dihes(mol):
    #pytmol label atom rank
    dihes=[]
    for bond in mol.GetBonds():
        if bond.GetBondType().name=='SINGLE' and bond.IsInRing()==False:#寻找不在环上的单键
            atom2id=bond.GetBeginAtomIdx()
            atom3id=bond.GetEndAtomIdx()
            atom2=bond.GetBeginAtom()
            atom3=bond.GetEndAtom()

            if len(atom3.GetNeighbors())==1  or len(atom2.GetNeighbors())==1 :
                pass
            else:
                atom1s=atom2.GetNeighbors()
                atom1sid=[ at.GetIdx()   for at in atom1s]
    #             print(atom1sid,atom3id)
                atom1sid.remove(atom3id)
                atom1id=atom1sid[0]

                atom4s=atom3.GetNeighbors()
                atom4sid=[ at.GetIdx()   for at in atom4s]
    #             print(atom1sid,atom3id)
                atom4sid.remove(atom2id)
                atom4id=atom4sid[0]
#                 print(atom1id,atom2id,atom3id,atom4id)
                dihe=[atom1id,atom2id,atom3id,atom4id]
                dihes.append(dihe)
    return dihes

def get_path_atoms(m,Mark1,Mark2):
    start_atom_index = Get_id_bysymbol(m,Mark1) # 第一个碳原子
    target_atom_index = Get_id_bysymbol(m,Mark2)  # 氧原子，索引为 -1 表示最后一个原子
    graph=get_graph(m)
    # 获取所有路径
    paths = dfs_paths(graph, start_atom_index, target_atom_index)
    ATOMS=[]
    for path in paths:
        #print(path)
        for atom in path:
            if atom not in ATOMS:
                ATOMS.append(atom)
    return ATOMS# 输出所有路径
    
def get_Nei_group(m,M1,M2):
    ssr = Chem.GetSymmSSSR(m)
    RING_Atom=[]
    idx=get_path_atoms(m,M1,M2)
    for r_L in ssr:
        RING_Atom.append(list(r_L))
    idx2=[]#获取相邻环
    for bond in m.GetBonds():
        atom1 = bond.GetBeginAtomIdx()
        atom2 = bond.GetEndAtomIdx()
        for i in range(len(RING_Atom)):
            if (atom1 in idx and atom2 in RING_Atom[i]) or (atom2 in idx and atom1 in RING_Atom[i]):
                for j in range(len(RING_Atom[i])):
                    idx2.append(RING_Atom[i][j])
            else:
                pass
    idx3=[]
    for i in range(len(RING_Atom)):
        set_c = set(RING_Atom[i]) & set(idx2)
        list_c = list(set_c)
        if list_c and list_c!=RING_Atom[i]:
            for j in range(len(RING_Atom[i])):
                idx3.append(RING_Atom[i][j])
    IDX=[]
    for id in [idx,idx2,idx3]:
        for j in id:
            if j not in IDX:
                IDX.append(j)
    #print(IDX)
    IDX_atoms=[m.GetAtomWithIdx(x) for x in IDX]#周围官能团：
    IDX2=[x for x in IDX]
    for atom in IDX_atoms:
        for atom2 in atom.GetNeighbors():
            if atom2.GetIdx() not in IDX:
                IDX.append(atom2.GetIdx())
            for atom3 in atom2.GetNeighbors():
                if atom3.GetSymbol()!='C':
                    IDX.append(atom3.GetIdx())
    for atom in IDX_atoms:
        for atom2 in atom.GetNeighbors():
            if atom2.GetIdx() not in IDX2 and atom2.GetSymbol()!='C':
                IDX2.append(atom2.GetIdx())
            nei=[]
            for nei1 in atom2.GetNeighbors():
                #print(nei1 in IDX_atoms,nei1.GetIdx())
                if nei1.GetIdx() not in [a.GetIdx() for a in IDX_atoms]:
                    nei.append(nei1)
            #print([atom.GetIdx() for atom in nei])
            if atom2.GetIdx() not in IDX2 and atom2.GetSymbol()=='C' and all(atom3.GetSymbol()!='C' for atom3 in nei):
                IDX2.append(atom2.GetIdx())
                #print('zz',atom2.GetIdx())
                for atom3 in atom2.GetNeighbors():
                    IDX2.append(atom3.GetIdx())
                     #print(atom3.GetIdx())
    IDX3=[]
    IDX4=[]
    for i in range(len(IDX)):
        if IDX[i] not in IDX3:
            IDX3.append(IDX[i])
    for i in range(len(IDX2)):
        if IDX2[i] not in IDX4:
            IDX4.append(IDX2[i])            
    return IDX3,IDX4#前面为包含亚甲基，后面为不包含亚甲基的部分

def Decomp(m,M1,M2):
    IDX,IDX2=get_Nei_group(m,M1,M2)
    BB=Chem.MolFromSmiles(Chem.MolToSmiles(Get_Mol(m,IDX2)))#取代基位置的根据化合价自动加氢
    try:
        SC=Chem.GetMolFrags(AllChem.ReplaceCore(m, Get_Mol(m,IDX2)),asMols=True)
    except:
        SC=[]
    return BB,SC

def dfs_paths(graph, start, end, path=[]):#从start到end的所有路径
    path = path + [start]
    if start == end:
        return [path]
    if start not in graph:
        return []
    paths = []
    for atom in graph[start]:
        if atom not in path:
            new_paths = dfs_paths(graph, atom, end, path)
            for new_path in new_paths:
                paths.append(new_path)
    return paths

def get_graph(mol):#获取分子连接
    graph={}
    for bond in mol.GetBonds():
        atom1 = bond.GetBeginAtomIdx()
        atom2 = bond.GetEndAtomIdx()
        if atom1 not in graph:
            graph[atom1] = []
        if atom2 not in graph:
            graph[atom2] = []
        graph[atom1].append(atom2)
        graph[atom2].append(atom1)
    return graph

def Get_Not_C(mol):#获取侧链非碳原子数目
    N=0
    for atom in mol.GetAtoms():
        if atom.GetSymbol()!='C' and atom.GetSymbol()!='*':
            N+=1
    return N

def Get_Bifurcation_site(mol,mark):#获取分叉位点
    BEG_IDX=Get_id_bysymbol(mol,mark)
    #print(BEG_IDX)
    Len=[]
    for atom in mol.GetAtoms():
    # 检查当前原子是否为碳原子
        if atom.GetAtomicNum() == 6:  # 6 是碳原子的原子序数
            # 检查当前碳原子是否连接到至少一个非氢原子并且是SP3杂化
            if atom.GetTotalNumHs()==1 and atom.GetHybridization()==Chem.HybridizationType.SP3:
                BS=atom.GetIdx()
                path=rdmolops.GetShortestPath(mol, BEG_IDX, BS)
                Len.append(len(path)-1)
    return Len

def Get_SC_INF(m,M1,M2):
    BB,SC=Decomp(m,M1,M2)
    if SC:
        SMI=[Chem.MolToSmiles(x) for x in SC]
        SMI2=[]
        print(SMI)
        for smi in SMI:
            stri=re.match('\[(\d+)\*\]',smi)[0]
            smi=smi.replace(stri,'[*]')
            SMI2.append(smi)
        _1,smi=unique_molecules(SMI2)#侧链去重，获取侧链种类
        SC=[Chem.MolFromSmiles(x) for x in smi]
        INFO=[]
        #print(smi)
        for i in range(len(SC)):
            #mark=re.match(r'\[\d+\*\]', Chem.MolToSmiles(SC[i]))[0]
            N=Get_Not_C(SC[i])
            Len=Get_Bifurcation_site(SC[i],'*')
            if len(Len)!=0:
                INFO.append([len(Len),Len[0],N,SC[i].GetNumAtoms()])
            else:
                INFO.append([0,0,N,SC[i].GetNumAtoms()])
        a=np.sort(np.array([x[1] for x in INFO]))[::-1]#排序按照分叉位点从多到少
        BS=[0,0]#只考虑分叉位点前两名的侧链。为0表示侧链没有分叉位点，为+N表示侧链在N号位置分叉，为-1表示没有侧链。
        for i in range(len(a)):
            try:
                BS[i]=a[i]
            except:
                pass
    else:
        BS=[-1,-1]
    return BS

def Get_N_COS2(COS2,N):#获取分子总长为N的COS2序列
    COS2_2=[]
    for i in range(len(COS2)):
        a=[]
        for j in range(N):
            a.append(COS2[i][j%len(COS2[i])])
        COS2_2.append(a)
    return COS2_2

def Invert_COS2(sequences,N):
# 对每个序列进行排序
    sorted_sequences = [sorted(seq) for seq in sequences]
    
    # 创建一个词汇表，包含所有序列中的唯一元素
    vocab = set()
    for seq in sorted_sequences:
        vocab.update(seq)
    
    # 将每个元素映射到一个唯一的索引
    vocab_to_index = {token: idx + 1 for idx, token in enumerate(vocab)}
    vocab_size = len(vocab_to_index)
    
    # 将序列转换为索引列表
    indexed_sequences = [[vocab_to_index[token] for token in seq] for seq in sorted_sequences]
    # 计算嵌入表示
    embedding_layer = nn.Embedding(num_embeddings=vocab_size + 1, embedding_dim=N)
    
    # 计算每个序列的嵌入表示
    embedded_sequences = []
    for seq in indexed_sequences:
        seq_tensor = torch.tensor(seq)
        embedded_seq = embedding_layer(seq_tensor)
        # 对每个序列中的元素进行求和得到排序后的嵌入表示
        summed_seq = torch.sum(embedded_seq, dim=0)
        embedded_sequences.append(summed_seq)

    # 转换为 PyTorch Tensor
    embedded_sequences_tensor = torch.stack(embedded_sequences)
    return embedded_sequences_tensor.detach().numpy()
    
def Get_2U_COS2(file,N,module,Invert=None):#从源文件获取特征和目标信息，包括HOMO，LUMO，Mn，以及PDI和COS2特征，只保留HP聚合物
    if isinstance(file,pd.DataFrame):
        data2=file
    else:
        data2=pd.read_csv(file)
    smi1=Process_smi(data2['dimers1'].values)
    SMI1=[x for a in smi1 for x in a]
    SMI=SMI1
    U_MOL,U_SMI=unique_molecules(SMI)#去重
    print(f'Collect 2Unit:{len(U_SMI)}')
    pred_y=SMI_Prediction(U_SMI,"mob_smi.csv",module)
    bb1=Get_U_COS2(smi1,U_SMI,pred_y)
    if Invert=='Embedded':
        COS2=Invert_COS2(bb1,N)
        return COS2
    if Invert==None:
        COS2=Get_N_COS2(bb1,N)
        return COS2
    if Invert=='Extended':
        smi3=[]
        idx=[]
        for i in range(len(smi1)):
            a=[]
            for j in range(len(smi1[i])):
                if smi1[i][j:]+smi1[i][:j] not in a:
                    a.append(smi1[i][j:]+smi1[i][:j])
                    smi3.append(smi1[i][j:]+smi1[i][:j])
                    idx.append(i)
                if smi1[i][::-1][j:]+smi1[i][::-1][:j] not in a:
                    a.append(smi1[i][::-1][j:]+smi1[i][::-1][:j])
                    smi3.append(smi1[i][::-1][j:]+smi1[i][::-1][:j])
                    idx.append(i)
        bb1=Get_U_COS2(smi3,U_SMI,pred_y)
        COS2=Get_N_COS2(bb1,N)
        return COS2,idx
    if Invert=='Sorted':
        smi4=[]
        for i in range(len(smi1)):
            Atom_N=[Chem.MolFromSmiles(x).GetNumAtoms() for x in smi1[i]]
            sorted_id = sorted(range(len(Atom_N)), key=lambda k: Atom_N[k], reverse=True)
            a=[ii for ii in sorted_id]
            for j in range(len(smi1[i])):
                a[j]=smi1[i][sorted_id.index(j)]
                
            smi4.append(a)
        bb1=Get_U_COS2(smi4,U_SMI,pred_y)
        COS2=Get_N_COS2(bb1,N)
        return COS2
        
def Get_Feature(COS2,data2,features1,SC=0):
    idx=[]
    for i in range(data2.shape[0]):
        condition=[pd.isna(data2[feature][i])==False for feature in features1]
        if all(condition) and data2['class'][i]=='HP':
            idx.append(i)
    if SC!=0:
        features2=['SC_INF1','SC_INF2']
    else:
        features2=[]
    FT=pd.DataFrame(index=[i for i in range(len(idx))],columns=['Indices','smiles']+features1+features2+[f'COS2_{i}' for i in range(len(COS2[0]))]+['u_h','u_e'])
    print(f'Get Infomation:{FT.keys()},Number:{FT.shape[0]}')
    for i in range(len(idx)):
        I=idx[i]
        if 'poly1' in data2.columns:
            FT['smiles'][i]=data2['poly1'][I]
        else:
            FT['smiles'][i]=data2['smiles'][I]
        FT['Indices'][i]=I
        for j in range(len(COS2[0])):
            FT[f'COS2_{j}'][i]=COS2[I][j]
        FT['u_h'][i]=data2['hole_mobility'][I]
        FT['u_e'][i]=data2['electron_mobility'][I]
        for feature in features1:
            try:
                FT[feature][i]=data2[feature][I]
            except:
                print(f'{feature} is not exist in orignal data')
        if features2:
            try:
                m=Chem.MolFromSmiles(FT['smiles'][i],sanitize=False)#侧链信息获取必须是凯库勒形式
                SC=Get_SC_INF(m,'Fr','Cs')
                FT['SC_INF1'][i],FT['SC_INF2'][i]=SC[0],SC[1]
            except:
                smi=FT['smiles'][i]
                print(f'SideChain informatin of polymer:{smi}\t{i} get failed')
    return FT

def Process_FT(FT):#特征处理，分子去重
    N=[]
    for i in range(FT.shape[0]):
        if FT.iloc[i,FT.shape[1]-1]!=0 or FT.iloc[i,FT.shape[1]-2]!=0:
            N.append(i)
    list_smi=[FT['smiles'][i].replace('Fr','H').replace('Cs','H') for i in N]
    mol,smi=unique_molecules(list_smi)
    idx={}
    for i in N:
        mol = Chem.RemoveHs(Chem.MolFromSmiles(FT['smiles'][i].replace('Fr','H').replace('Cs','H')))
        if mol is not None:
            canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
            idxx=smi.index(canonical_smiles)
            if idxx not in idx:
                idx[idxx]=[i]
            else:
                idx[idxx].append(i)
    FT2=pd.DataFrame(index=[i for i in range(len(smi))],columns=FT.columns)
    for key,value in idx.items():
        if len(value)>1:
            FT2['Indices'][key]=value
            FT2['smiles'][key]=smi[key]
            for j in range(2,FT.shape[1]):
                FT2.iloc[key,j]=sum(FT.iloc[value,j])/len(value)
        else:
            FT2.iloc[key,:]=FT.iloc[value,:]
    return FT2.iloc[:,2:(FT.shape[1]-2)].values.tolist(),FT2['u_h'].values.tolist(),FT2['u_e'].values.tolist(),FT2

def train_test_split(random_seed,ratio,x,y,log=True):
    data_array=np.arange(0,len(x),1)
    np.random.seed(random_seed)
    np.random.shuffle(data_array)
    train_x,train_y,test_x,test_y=[],[],[],[]
    train_L=int(ratio*len(data_array))
    N=[]
    if log:
        for i in data_array[:train_L]:
            if y[i]!=0:
                train_y.append(np.log(y[i]))
                train_x.append(x[i])
                N.append(i)
        for i in data_array[train_L:]:
            if y[i]!=0:
                test_y.append(np.log(y[i]))
                test_x.append(x[i])
                N.append(i)
    else:
        for i in data_array[:train_L]:
            train_y.append(y[i])
            train_x.append(x[i])
            N.append(i)
        for i in data_array[train_L:]:
            test_y.append(y[i])
            test_x.append(x[i])
            N.append(i)
    return train_x,train_y,test_x,test_y,N

def Train_data(train_x,train_y,test_x,test_y):
    rs=42
    regressors = [
        ('Lasso Regressor',Lasso(random_state=rs)),
        ('RlasticNet Regressor',ElasticNet(random_state=rs)),#random_state=rs
        ('PolynomilaFeatures',make_pipeline(PolynomialFeatures(4), LinearRegression())),
        ("Linear Regression", LinearRegression()), # 线性回归模型
        ("Ridge Regression", Ridge(random_state=rs)), # 岭回归模型
        ("Support Vector", SVR()),  # 支持向量回归模型
        ("K-Nearest Neighbors", KNeighborsRegressor()),  # K-最近邻回归模型
        ("Decision Tree", DecisionTreeRegressor(random_state=rs)),  # 决策树回归模型
        ("Random Forest", RandomForestRegressor(random_state=rs)), # 随机森林回归模型
        ("Gradient Boosting", GradientBoostingRegressor(random_state=rs)), # 梯度提升回归模型
        ("XGBoost", XGBRegressor(random_state=rs)), # XGBoost回归模型
        ("LightGBM", LGBMRegressor(random_state=rs)), # LightGBM回归模型
        ("Multi-layer Perceptron", MLPRegressor( # 多层感知器（神经网络）回归模型
            hidden_layer_sizes=(128,64,32),
            learning_rate_init=0.0001,
            activation='relu', solver='adam',
            batch_size=16,
            max_iter=10000, random_state=rs)),
    ]
    R2={}
    Pred_y={}
    for name, regressor in regressors:
        regressor.fit(train_x,train_y)
        pred_train_y = regressor.predict(train_x)
        pred_test_y = regressor.predict(test_x)
        SSE=np.sum((pred_test_y-test_y)**2)
        SST=np.sum((test_y-np.mean(test_y))**2)
        R2[name]=1-SSE/SST
        Pred_y[name]=pred_test_y
        #print(name,R2[name])
    return R2,Pred_y

def Get_train_data(collect_file,N,model,features,Extend=True,COS2=None,SC=0):
    data2=pd.read_csv(collect_file)
    if not isinstance(COS2,dict):
        print('smiles_process')
        COS2=Get_2U_COS2(collect_file,N,model,None)
        FT=Get_Feature(COS2,data2,features,SC)
        x,y1,y2,FT2=Process_FT(FT)
        print('Collect data:',FT2.shape[0])
        if Extend:
            FT2['dimers1']=['CC' for i in range(FT2.shape[0])]
            FT2['dimers2']=['CC' for i in range(FT2.shape[0])]
            for i in range(FT2.shape[0]):
                if isinstance(FT2['Indices'][i],int):
                    FT2['dimers1'][i]=data2['dimers1'][FT2['Indices'][i]]
                    FT2['dimers2'][i]=data2['dimers2'][FT2['Indices'][i]]
                else:
                    FT2['dimers1'][i]=data2['dimers1'][FT2['Indices'][i][0]]
                    FT2['dimers2'][i]=data2['dimers2'][FT2['Indices'][i][0]]        
            
            COS2_2,idx=Get_2U_COS2(FT2,N,model,'Extended')
            x=[]
            y1=[]
            y2=[]
            for i in range(len(COS2_2)):
                a=[]
                for fea in features:
                    a.append(FT2[fea][idx[i]])
                if SC!=0:
                    a.append(FT2['SC_INF1'][idx[i]])
                    a.append(FT2['SC_INF2'][idx[i]])
                a+=COS2_2[i]
                x.append(a)
                y1.append(FT2['u_h'][idx[i]])
                y2.append(FT2['u_e'][idx[i]])
        else:
            pass
    else:
        print('Atom_process')
        FT=Get_Feature(COS2['COS2'],data2,features,SC)
        x,y1,y2,FT2=Process_FT(FT)
        if Extend:
            FT2['dimers1']=['CC' for i in range(FT2.shape[0])]
            FT2['dimers2']=['CC' for i in range(FT2.shape[0])]
            for i in range(FT2.shape[0]):
                if isinstance(FT2['Indices'][i],int):
                    FT2['dimers1'][i]=data2['dimers1'][FT2['Indices'][i]]
                    FT2['dimers2'][i]=data2['dimers2'][FT2['Indices'][i]]
                else:
                    FT2['dimers1'][i]=data2['dimers1'][FT2['Indices'][i][0]]
                    FT2['dimers2'][i]=data2['dimers2'][FT2['Indices'][i][0]]        
            
            smi1=Process_smi(FT2['dimers1'].values)
            smi3=[]
            idx=[]
            for i in range(len(smi1)):
                a=[]
                for j in range(len(smi1[i])):
                    if smi1[i][j:]+smi1[i][:j] not in a:
                        a.append(smi1[i][j:]+smi1[i][:j])
                        smi3.append(smi1[i][j:]+smi1[i][:j])
                        idx.append(i)
                    if smi1[i][::-1][j:]+smi1[i][::-1][:j] not in a:
                        a.append(smi1[i][::-1][j:]+smi1[i][::-1][:j])
                        smi3.append(smi1[i][::-1][j:]+smi1[i][::-1][:j])
                        idx.append(i)
            SMI=[]
            for i in range(len(smi3)):
                a=[]
                for j in range(len(smi3[i])):
                    a.append(Chem.MolToSmiles(Chem.RemoveHs(Chem.AddHs(Chem.MolFromSmiles(smi3[i][j])))))
                SMI.append(a)
            bb1=Get_U_COS2(SMI,COS2['Unit'],COS2['Pred_E'])
            COS2_2=Get_N_COS2(bb1,N)
            x=[]
            y1=[]
            y2=[]
            for i in range(len(COS2_2)):
                a=[]
                for fea in features:
                    a.append(FT2[fea][idx[i]])
                if SC!=0:
                    a.append(FT2['SC_INF1'][idx[i]])
                    a.append(FT2['SC_INF2'][idx[i]])
                a+=COS2_2[i]
                x.append(a)
                y1.append(FT2['u_h'][idx[i]])
                y2.append(FT2['u_e'][idx[i]])
        else:
            pass
    
    return x,y1,y2
    
def unique_acc_smile(smi_list,data3):
    uniq_smi=unique_molecules(smi_list)[1]
    oect_v=[]
    dimer=[]
    idx=[]
    for i in range(len(uniq_smi)):
        a=np.array([0.0,0.0,0.0,0.0,0.0],dtype=np.float64)
        N=0
        for j in range(data3.shape[0]):
            smi=Chem.MolToSmiles(Chem.MolFromSmiles(data3['smiles'][j].replace('Fr','H').replace('Cs','H')))
            if smi==uniq_smi[i]:
                a+=data3.iloc[j,1:6].values.astype(np.float64)
                N+=1
                di=data3['dimers1'][j]
                
        idx.append(data3['dimers1'].values.tolist().index(di))
        dimer.append(di)
        
        oect_v.append([data3['smiles'][i]]+list(a/N))
    data3=pd.DataFrame(oect_v)
    
    data3.columns=['smiles','HOMO(eV)','LUMO(eV)','hole mobility','electron mobility','uC*']
    data3['dimers1']=dimer
    data3['Index']=idx
    return data3

def Plot_XY(x,y,color,size,x_min=None,x_max=None,grid=True,tick=None,diag=True,diag_line='dashed',save_name=None,other_plot_dashed=None,other_plot_solid=None):
    plt.style.use('fast')
    plt.figure(figsize=(9,9))
    plt.scatter(x,y,c=color,s=size)
    if diag:
        plt.plot([-100,100],[-100,100],color='black',linestyle=diag_line)
    plt.tick_params(axis='x',labelsize=25,colors='black')
    plt.tick_params(axis='y',labelsize=25,colors='black')
    MIN=min([min(x),min(y)])
    MAX=max([max(x),max(y)])
    if other_plot_dashed is not None:
        for x in other_plot_dashed:
            plt.plot(x[0],x[1],c='black',linestyle='dashed')
    if other_plot_solid is not None:
        for x in other_plot_solid:
            plt.plot(x[0],x[1],c='black',linestyle='solid')            
    if not x_min:
        plt.xlim(round(MIN-(MAX-MIN)*0.1),round(MAX+(MAX-MIN)*0.1))
        plt.ylim(round(MIN-(MAX-MIN)*0.1),round(MAX+(MAX-MIN)*0.1))
    if x_min:
        plt.xlim(x_min,x_max)
        plt.ylim(x_min,x_max)
    if tick is None:
        tick=np.linspace(round(MIN),round(MAX),5)
    plt.xticks(tick)
    plt.yticks(tick)
    plt.grid(grid)
    if save_name:
        plt.savefig(save_name)
    plt.show()


def OECT_data(model1,model2,target='mobility'):
    data3=pd.read_csv('OECT.csv')
    smi_list=[x.replace('Fr','H').replace('Cs','H') for x in data3['smiles'].values.tolist()]
    data3=unique_acc_smile(smi_list,data3)
    dimer=data3['dimers1'].values
    smi1=Process_smi(dimer)
    U_SMI=[]
    for i in range(len(smi1)):
        for j in range(len(smi1[i])):
            if smi1[i][j] not in U_SMI:
                U_SMI.append(smi1[i][j])
    U_MOL,U_SMI=unique_molecules(U_SMI)#去重
    pred_y=SMI_Prediction(U_SMI,"OECT_smi.csv",'./Unimolsave')
    
    
    oect=np.load('OECT.npy',allow_pickle=True).item()
    delta_H=oect['delta_H']
    delta_L=oect['delta_L']
    smi2=[]
    idx=[]
    d_H=[]
    d_L=[]
    extend=True
    for i in range(len(smi1)):
        a=data3['Index'][i]
        if extend:
            for j in range(len(smi1[i])):
                if smi1[i][j:]+smi1[i][:j] not in smi2:
                    smi2.append(smi1[i][j:]+smi1[i][:j])
                    d_H.append(delta_H[a][j:]+delta_H[a][:j])
                    d_L.append(delta_L[a][j:]+delta_L[a][:j])
                    idx.append(i)
                if smi1[i][::-1][j:]+smi1[i][::-1][:j] not in smi2:
                    smi2.append(smi1[i][::-1][j:]+smi1[i][::-1][:j])
                    d_H.append(delta_H[a][::-1][j:]+delta_H[a][::-1][:j])
                    d_L.append(delta_L[a][::-1][j:]+delta_L[a][::-1][:j])
                    idx.append(i)
        else:
            smi2.append(smi1[i])
            d_H.append(delta_H[a])
            d_L.append(delta_L[a])
            idx.append(i)
    bb1=Get_U_COS2(smi2,U_SMI,pred_y)
    COS2=Get_N_COS2(bb1,10)
    OECT_T=[]
    oect_x2=[]
    ue=[]
    OECT_x2=[]
    uc2=[]
    OECT_x1=[]
    oect_x1=[]
    uh=[]
    uc1=[]
    Ce=[]
    Ch=[]
    for i in range(len(COS2)):
        if (not pd.isna(data3['HOMO(eV)'][idx[i]])) and (not pd.isna(data3['LUMO(eV)'][idx[i]])):
            a=[]
            a+=[data3['HOMO(eV)'][idx[i]]]
            a+=[data3['LUMO(eV)'][idx[i]]]
            a+=COS2[i]
            
            a+=[np.mean(d_H[i])]
            a+=[np.sqrt((sum((d_H[i] - np.mean(d_H[i])) ** 2)).mean())]
            a+=[np.mean(d_L[i])]
            a+=[np.sqrt((sum((d_L[i] - np.mean(d_L[i])) ** 2)).mean())]
            if not pd.isna(data3['electron mobility'][idx[i]]) and data3['electron mobility'][idx[i]]!=0:
                oect_x2.append(a)
                #print(data3['uC*'][idx[i]])
                ue.append(np.log(data3['electron mobility'][idx[i]]))
                if not pd.isna(data3['uC*'][idx[i]]):
                    OECT_x2.append(a)
                    #print(COS2[i],i,idx[i])
                    uc2.append(np.log(data3['uC*'][idx[i]]))
                    Ce.append(data3['uC*'][idx[i]]/data3['electron mobility'][idx[i]])
                    #print('e',data3['uC*'][idx[i]],data3['electron mobility'][idx[i]],data3['uC*'][idx[i]]/data3['electron mobility'][idx[i]],idx[i])
            if not pd.isna(data3['hole mobility'][idx[i]]) and data3['hole mobility'][idx[i]]!=0:
                oect_x1.append(a)
                #print(data3['uC*'][idx[i]])
                uh.append(np.log(data3['hole mobility'][idx[i]]))
                if not pd.isna(data3['uC*'][idx[i]]):
                    OECT_x1.append(a)
                    #print(COS2[i],i,idx[i])
                    uc1.append(np.log(data3['uC*'][idx[i]]))
                    Ch.append(data3['uC*'][idx[i]]/data3['hole mobility'][idx[i]])
                    #print(data3['uC*'][idx[i]],data3['hole mobility'][idx[i]],data3['uC*'][idx[i]]/data3['hole mobility'][idx[i]],idx[i])
    if target=='mobility':
        oect_y1=model1.predict([x[:12] for x in oect_x1])
        oect_x12,h=[[oect_y1[i]]+oect_x1[i][12:]+oect_x1[i][:2] for i in range(len(oect_x1))],[uh[i] for i in range(len(oect_x1))]
        oect_y2=model2.predict([x[:12] for x in oect_x2])
        oect_x22,e=[[oect_y2[i]]+oect_x2[i][12:]+oect_x2[i][:2] for i in range(len(oect_x2))],[ue[i] for i in range(len(oect_x2))]
    else:
        oect_y1=model1.predict([x[:12] for x in OECT_x1])
        oect_x12,h=[[oect_y1[i]]+OECT_x1[i][12:]+OECT_x1[i][:2] for i in range(len(OECT_x1))],[uc1[i] for i in range(len(OECT_x1))]
        oect_y2=model2.predict([x[:12] for x in OECT_x2])
        oect_x22,e=[[oect_y2[i]]+OECT_x2[i][12:]+OECT_x2[i][:2] for i in range(len(OECT_x2))],[uc2[i] for i in range(len(OECT_x2))]
    return oect_x12,h,oect_x22,e

def screen_OECT(model1,model2):
    data=np.load('conf.npy',allow_pickle=True).item()
    DP=pd.DataFrame(data['feature'])
    DP.iloc[:,1]=DP.iloc[:,1].apply(lambda x:x-0.8)
    mob_ele=model1.predict(DP.iloc[:,:12])
    SMILES_ALL=[Chem.MolToSmiles(Chem.MolFromSmiles(x[0])) for x in data['smiles']]
    DD=[[mob_ele[i]]+DP.iloc[i,12:].values.tolist()+DP.iloc[i,:2].values.tolist() for i in range(DP.shape[0])]#range(DP.shape[0]) 
    mob_ele=np.zeros(len(DD))
    for mo in model2:
        mob_ele+=mo.predict(DD)/len(model2)
    mob=[]
    indices=np.argsort(mob_ele)[-len(DD):][::-1]
    ii=[]
    U=[]
    for x in indices[:200]:
        mob.append(np.exp(mob_ele[x]))
        print(DP.iloc[x,:],DD[x],np.exp(mob_ele[x]))
        ii.append(x)
        U.append(data['Unit_idx'][x][0])
        U.append(data['Unit_idx'][x][1])
    mols=[Chem.MolFromSmiles(data['smiles'][i][0]) for i in ii]
    U2={}
    for i in range(len(U)):
        if U[i] not in U2:
            U2[U[i]]=1
        else:
            U2[U[i]]+=1
    tim=[]
    n=[]
    for i in range(0,60):
        for key,value in U2.items():
            if value==i:
                n.append(key)
                tim.append(value)
    
    Unit=pd.read_csv('Units-2.csv')
    U_mols=[Chem.MolFromSmiles(Unit['smiles'][i]) for i in n[::-1]]
    Times=tim[::-1]
    return mols,mob,U_mols,Times
    
x1,y1,y2=Get_train_data('mob.csv',10,'./Unimolsave',['HOMO(eV)','LUMO(eV)','Mn(kg/mol)','PDI'],SC=1)
x2=[x[:2]+x[6:16] for x in x1]
train_x1,train_y1,test_x1,test_y1,N=train_test_split(964,0.9,x2,y1)
train_x2,train_y2,test_x2,test_y2,N=train_test_split(964,0.9,x2,y2)
model1=XGBRegressor(random_state=42)
model1.fit(train_x1,train_y1)
print(model1.score(test_x1,test_y1))
model2=XGBRegressor(random_state=42)
model2.fit(train_x2,train_y2)
print(model2.score(test_x2,test_y2))
x12,y12,x22,y22=OECT_data(model1,model2,target='uc')
train_x1,train_y1,test_x1,test_y1,N=train_test_split(405,0.9,x12,y12,False)
train_x2,train_y2,test_x2,test_y2,N=train_test_split(47,0.9,x22,y22,False)
model12=XGBRegressor(random_state=42)
model12.fit(train_x1,train_y1)
print(model12.score(test_x1,test_y1))
model22=XGBRegressor(random_state=42)
model22.fit(train_x2,train_y2)
print(model22.score(test_x2,test_y2))
a=np.random.choice(1000,50,False)
m1,m2=[],[]
for i in a:
    train_x1,train_y1,test_x1,test_y1,N=train_test_split(i,0.9,x12,y12,False)
    train_x2,train_y2,test_x2,test_y2,N=train_test_split(i,0.9,x22,y22,False)
    model12=XGBRegressor(random_state=42)
    model12.fit(train_x1,train_y1)
    if model12.score(test_x1,test_y1)>0.95 and len(m1)<5:
        m1.append(model12)
        print(model12.score(test_x1,test_y1))
    model22=XGBRegressor(random_state=42)
    model22.fit(train_x2,train_y2)
    if model22.score(test_x2,test_y2)>0.95 and len(m2)<5:
        m2.append(model22)
        print(model22.score(test_x2,test_y2))

'''   ''' 
mols,uc,U_mols,Times=screen_OECT(model2,m2)
#print(mob)
Draw.MolsToGridImage(mols[:40],molsPerRow=4, legends=[str(round(x,3)) for x in uc[:40]], subImgSize=(250,250),useSVG=True)
#print([Chem.MolToSmiles(x) for x in mols[:40]])
Draw.MolsToGridImage(U_mols[:20],molsPerRow=4, legends=[str(x) for x in Times[:20]], subImgSize=(250,250),useSVG=True)
mols,uc,U_mols,Times=screen_OECT(model1,m1)
#print(mob)
Draw.MolsToGridImage(mols[:40],molsPerRow=4, legends=[str(round(x,3)) for x in uc[:40]], subImgSize=(250,250),useSVG=True)
#print([Chem.MolToSmiles(x) for x in mols[:40]])
Draw.MolsToGridImage(U_mols[:20],molsPerRow=4, legends=[str(x) for x in Times[:20]], subImgSize=(250,250),useSVG=True)
