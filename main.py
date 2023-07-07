import numpy as np
from boundaries import apply_bc, create_load_bc
from classes import Frame, Element, Node
from mesh import mesh_sign
from plot_results import plot_output
from stiffness import q4_k_matrix, stress_strain_matrix, frame_k_matrix, assemble, trans_k_matrix, q4_b_matrix

STRESS_MODE = 0  # Plane Stress: 0 | Plane Strain: 1
m, n = 10, 8  # Number of elements in horizontal and vertical directions
P = -24  # N/m
SIGN_L, SIGN_W, SIGN_T = 2, 1, 0.02  # Length, Width and Thickness of the sign
FRAME_RADIUS = 15E-3
FRAME_AREA, FRAME_INERTIA = np.pi * (FRAME_RADIUS ** 2), np.pi * (FRAME_RADIUS ** 4) / 4  # Frame properties
E1, v1, E2, v2 = 200E9, 0.3, 210E9, 0.25  # Young's modulus and Poisson's ratio

# ◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯ The frame
# Defining frame nodes and elements
frame_nodes = {1: Node(number=1, position=(0, 0), force=(np.nan, np.nan, np.nan), displacement=(0, 0, 0)),
               2: Node(number=2, position=(0, 5), force=(0, 0, 0), displacement=(np.nan, np.nan, np.nan)),
               3: Node(number=3, position=(2 - SIGN_W / n, 5), force=(0, 0, 0), displacement=(np.nan, np.nan, np.nan))}
frame_elems = {1: Frame(number=1, nodes=(1, 2), young=E1, area=FRAME_AREA, inertia=FRAME_INERTIA),
               2: Frame(number=2, nodes=(2, 3), young=E1, area=FRAME_AREA, inertia=FRAME_INERTIA)}

# Calculating and assembling the stiffness matrix of the frame
frame_local_k = dict()
for e in frame_elems:
    frame_local_k[e] = frame_k_matrix(frame_elems[e], frame_nodes)

# ◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯ The sign
# Creating load boundary condition
load_bc = create_load_bc(start_position=(2, 5), l=SIGN_L, w=SIGN_W, m=m, P=P)
# Meshing the sign
n_start_num, e_start_num = len(frame_nodes) + 1, len(frame_elems) + 2
sign_nodes, sign_elems = mesh_sign(n_start_num, e_start_num, (2, 5), length=SIGN_L,
                                   width=SIGN_W, t=SIGN_T, m=m, n=n, bc=load_bc, E1=E1, v1=v1, E2=E2, v2=v2, s_mode=STRESS_MODE)

# Calculating and assembling the stiffness matrix of the sign
sign_local_k = dict()
for e in sign_elems.values():
    sign_local_k[e.number] = q4_k_matrix(e, sign_nodes)

# ◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯ The Transition element
trans_element = {e_start_num - 1: Element(number=e_start_num - 1, nodes=(3, 5, 4), young=E1)}
trans_k = {e_start_num - 1: trans_k_matrix(b=SIGN_W / n, h=SIGN_W / n, E=E1, v=v1, stress_mode=STRESS_MODE, thickness=SIGN_T)}

# ◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯◯ Global assembly
nodes = frame_nodes | sign_nodes
elements = frame_elems | trans_element | sign_elems
local_k = frame_local_k | trans_k | sign_local_k

dof = {num: len(single_node.displacement) for num, single_node in nodes.items()}  # Each node's degrees of freedom
diff = {m: sum(dof[i] for i in range(1, m)) for m in nodes}  # Previous degrees of freedom of each element

# Extracting displacement and force vectors
d, f = np.zeros(sum(dof.values())), np.zeros(sum(dof.values()))
for node in nodes:
    d[diff[node]:diff[node] + dof[node]] = nodes[node].displacement
    f[diff[node]:diff[node] + dof[node]] = nodes[node].force

k = assemble(elements, local_k, dof)  # Assembling the global stiffness matrix

d = apply_bc(k, d, f)  # Applying boundary conditions
f = k @ d  # Finding unknown forces using {F}=[K]{D}
f[abs(f) < 0.1] = 0  # Ignoring small forces

np.set_printoptions(precision=2)
Node_A_d = nodes[2].get_primary_variables(d, diff) * 1E3
print(f"Displacement of node A is: {Node_A_d[0:2]} mm and it's magnitude is: {np.hypot(Node_A_d[0], Node_A_d[1])} mm")
Node_B_d = [value for key, value in nodes.items() if value.position == (2, 5)][0].get_primary_variables(d, diff) * 1E3
print(f"Displacement of node B is: {Node_B_d} mm and it's magnitude is: {np.hypot(Node_B_d[0], Node_B_d[1])} mm")
Node_C_d = [value for key, value in nodes.items() if value.position == (4, 5)][0].get_primary_variables(d, diff) * 1E3
print(f"Displacement of node C is: {Node_C_d} mm and it's magnitude is: {np.hypot(Node_C_d[0], Node_C_d[1])} mm")
Node_D_d = [value for key, value in nodes.items() if value.position == (4, 4)][0].get_primary_variables(d, diff) * 1E3
print(f"Displacement of node D is: {Node_D_d} mm and it's magnitude is: {np.hypot(Node_D_d[0], Node_D_d[1])} mm")


def q4_element_stress(D, element, all_nodes, element_d, s, t):
    X, Y = element.get_xy(all_nodes)
    B = q4_b_matrix(X, Y, s, t)
    return np.asarray(D @ B @ np.array(element_d)).reshape(-1)


c_element = elements[e_start_num]
elem_d = c_element.get_primary_variables(d, diff)
stress = q4_element_stress(stress_strain_matrix(STRESS_MODE, E1, v1), c_element,
                           nodes, elem_d, np.sqrt(3) / 3, np.sqrt(3) / 3)
print(f"Stress in node B is: {stress * 1E-3} kPa")

G = np.linalg.inv(k) @ f

plot_output(nodes, elements, np.delete(d, [2, 5, 8]), exaggeration=1)
