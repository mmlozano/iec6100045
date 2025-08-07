import numpy as np
import sympy as sp
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.optimize import minimize
from scipy.optimize import fsolve

# Definición de los parámetros
Vox = 1
R1x = 20.2
R2x = 0.814
R3x = 26.1
Lx = 10.9e-6
C = 6e-6
Zopt=2

# Definir el tiempo simbólico
t = sp.symbols('t')


# Definir la función tension-ca(t)
def tensionca(x, Vot, R1t, R2t, R3t, Lt):
    expr = 2 * R3t * Vot * sp.exp(-t * (C*R1t*R2t + C*R1t*R3t + Lt)/(2*C*Lt*R1t)) * \
           sp.sin(t * sp.sqrt((-C**2*R1t**2*(R2t**2 + 2*R2t*R3t + R3t**2) + 4*C*Lt*R1t**2 + 
                               2*C*Lt*R1t*(R2t + R3t) - Lt**2)/(C**2*Lt**2*R1t**2))/2)  / \
           (Lt*sp.sqrt((-C**2*R1t**2*(R2t**2 + 2*R2t*R3t + R3t**2) + 4*C*Lt*R1t**2 + 
                       2*C*Lt*R1t*(R2t + R3t) - Lt**2)/(C**2*Lt**2*R1t**2)))
    expr2 = sp.lambdify(t, expr, modules=['numpy'])
    return expr2(x)

# Definir la función corriente-cc(t)
def corrcc(x, Vot, R1t, R2t, R3t, Lt):
    exprc = 2 * Vot * sp.exp(-t * (R2t/Lt + 1/(C * R1t)) / 2) * sp.sin( t * sp.sqrt(-R2t**2/Lt**2 + 4/(C * Lt) + 2 * R2t/(C * Lt * R1t) - 1/(C**2 * R1t**2))/2)/(Lt * sp.sqrt(-R2t**2/Lt**2 + 4/(C*Lt) + 2*R2t/(C*Lt*R1t) - 1/(C**2*R1t**2))) 
    exprc2 = sp.lambdify(t, exprc, modules=['numpy'])
    return exprc2(x)

def corrcc_no_negatives(x, Vot, R1t, R2t, R3t, Lt):
    # Evaluamos la función original
    result = corrcc(x, Vot, R1t, R2t, R3t, Lt)
    # Inicializamos un array para almacenar los valores positivos
    result_positive = np.zeros_like(result)
    # Iteramos sobre los valores de la función original
    for i in range(len(result)):
        # Actualizamos los valores negativos a cero
        result_positive[i] = max(result[i], 0)
    return result_positive

# Definir la función refu(t)
def refu_t():
    return 1.037 * 1 * (1 - sp.exp(-t / (0.4074e-6))) * sp.exp(-t / (68.22e-6))

# Convertir la función simbólica en una función numpy para su evaluación
refu_t_np = sp.lambdify(t, refu_t(), modules=['numpy'])

# Definir la función refi(t)
def refi_t():
    return 0.01243 * 1e18 * 1 * 0.5 * (t+1.1e-6)**3 * sp.exp(-(t+1.1e-6)/(3.911*1e-6))

# Convertir la función simbólica en una función numpy para su evaluación
refi_t_np = sp.lambdify(t, refi_t(), modules=['numpy'])

# Generar datos de tiempo
t_data = np.linspace(1e-7, 100e-6, num=6000)


# Evaluar las funciones en los datos de tiempo
refu_t_eval = refu_t_np(t_data)
refi_t_eval = refi_t_np(t_data)
# Y las eñales evaluadas en los valores iniciales de la iteracion
tensionca_t_eval = tensionca(t_data, Vox, R1x, R2x, R3x, Lx)
corrcc_t_eval = corrcc(t_data, Vox, R1x, R2x, R3x, Lx)
corrcc_no_neg_t_eval = corrcc_no_negatives(t_data, Vox, R1x, R2x, R3x, Lx)

# Graficar antes de la optimización
plt.ion()
plt.figure(figsize=(10, 6))
plt.plot(t_data, tensionca_t_eval, label='u_eval(t)')
plt.plot(t_data, corrcc_t_eval, label='i_eval(t)')
plt.plot(t_data, corrcc_no_neg_t_eval, label='i_no_neg_eval(t)')
plt.plot(t_data, refu_t_eval, label='u_ref(t)')
plt.plot(t_data, refi_t_eval, label='i_ref(t)')
plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud')
plt.legend()
plt.grid(True)
plt.title('Antes de la optimización')
plt.show()

# Valores iniciales para los parámetros
#parametros_iniciales = [Vox, R1x, R2x, R3x, Lx]


def tf_u(variables):
    Vo, R1, R2, R3, L = variables
    umax=max(tensionca(t_data, Vo, R1, R2, R3, L))
    indice_t_30= np.argmax(tensionca(t_data, Vo, R1, R2, R3, L) >= (0.3 * umax))
    t_u_30=t_data[indice_t_30]
    indice_t_90= np.argmax(tensionca(t_data, Vo, R1, R2, R3, L) >= (0.9 * umax))
    t_u_90=t_data[indice_t_90] 
    return 1.67 * (t_u_90 - t_u_30)   

def tc_u(variables):
    Vo, R1, R2, R3, L = variables
    umax = max(tensionca(t_data, Vo, R1, R2, R3, L))
    #print("umax: ", umax)
    indice_t_501 = np.argmax(tensionca(t_data, Vo, R1, R2, R3, L) >= (0.5 * umax))
    t_u_501 = t_data[indice_t_501]
    #print("t_501: ", t_u_501)
    u501 = tensionca(t_u_501, Vo, R1, R2, R3, L)
    #print("u_501: ", u501)
    t_data2 = np.linspace(t_u_501+1e-6, 100e-6, num=6000)
    indice_t_502 = np.argmax(tensionca(t_data2, Vo, R1, R2, R3, L) <= (0.5 * umax))
    t_u_502 = t_data2[indice_t_502] 
    #print("t_u_502", t_u_502)
    u502 = tensionca(t_u_502, Vo, R1, R2, R3, L)
    #print("u_502: ", u502)
    return (t_u_502-t_u_501)

#tiempo_semiamp = tc_u((Vox, R1x, R2x, R3x, Lx))
#print("tcu: ", tiempo_semiamp)

def tc_i(variables):
    Vo, R1, R2, R3, L = variables
    imax = max(corrcc(t_data, Vo, R1, R2, R3, L))
    #print("imax: ", imax)
    indice_t_501 = np.argmax(corrcc(t_data, Vo, R1, R2, R3, L) >= (0.5 * imax))
    t_i_501 = t_data[indice_t_501]
    #print("t_501: ", t_i_501)
    i501 = corrcc(t_i_501, Vo, R1, R2, R3, L)
    #print("i_501: ", i501)
    t_data2 = np.linspace(t_i_501, 100e-6, num=6000)
    indice_t_502 = np.argmax(corrcc(t_data2, Vo, R1, R2, R3, L) <= (0.5 * imax))
    t_i_502 = t_data2[indice_t_502] 
    #print("t_i_502", t_i_502)
    i502 = corrcc(t_i_502, Vo, R1, R2, R3, L)
    #print("i_502: ", i502)
    return 1.18*(t_i_502-t_i_501)


def penalizacion_tc_u(variables):
    Vo, R1, R2, R3, L = variables
    tinf=40e-6
    tsup=60e-6
    tiempo=tc_u([Vo, R1, R2, R3, L])
    penalizacion_tcu = 0
    if tiempo < tinf:
        penalizacion_tcu = (tinf- tiempo)**2
    elif tiempo > tsup:
        penalizacion_u = (tsup- tiempo)**2
    return penalizacion_tcu


def penalizacion_tc_i(variables):
    Vo, R1, R2, R3, L = variables
    tinf=40e-6
    tsup=60e-6
    tiempo=tc_i([Vo, R1, R2, R3, L])
    penalizacion_tci = 0
    if tiempo < tinf:
        penalizacion_tci = (tinf- tiempo)**2
    elif tiempo > tsup:
        penalizacion_tci = (tsup- tiempo)**2
    return penalizacion_tci


def tf_i(variables):
    Vo, R1, R2, R3, L = variables
    imax=max(corrcc(t_data, Vo, R1, R2, R3, L))
    indice_t_10= np.argmax(corrcc(t_data, Vo, R1, R2, R3, L) >= (0.1 * imax))
    t_i_10=t_data[indice_t_10]
    indice_t_90= np.argmax(corrcc(t_data, Vo, R1, R2, R3, L) >= (0.9 * imax))
    t_i_90=t_data[indice_t_90] 
    return 1.25 * (t_i_90 - t_i_10)   

def zp(variables):
    Vo, R1, R2, R3, L = variables
    Zp= max(tensionca(t_data, Vo, R1, R2, R3, L))/max(corrcc(t_data, Vo, R1, R2, R3, L))
    return 1 * Zp

def penalizacion_tfu(variables):
    Vo, R1, R2, R3, L = variables
    tinf=0.87e-6
    tsup=1.56e-6
    tiempo=tf_u([Vo, R1, R2, R3, L])
    penalizacion_u = 0
    if tiempo < tinf:
        penalizacion_u = (tinf- tiempo)**2
    elif tiempo > tsup:
        penalizacion_u = (tsup- tiempo)**2

    return penalizacion_u

def penalizacion_tfi(variables):
    Vo, R1, R2, R3, L = variables
    tinf=6.4e-6
    tsup=9.6e-6
    tiempo=tf_i([Vo, R1, R2, R3, L])
    penalizacion_i = 0
    if tiempo < tinf:
        penalizacion_i = (tinf- tiempo)**2
    elif tiempo > tsup:
        penalizacion_i= (tsup- tiempo)**2

    return penalizacion_i

def penalizacion_z(variables):
    Vo, R1, R2, R3, L = variables
    zinf=1.6363
    zsup=2.4444
    z=zp([Vo, R1, R2, R3, L])
    penalizacion_z = 0
    if z < zinf:
        penalizacion_z = (zinf- z)**2
    elif z > zsup:
        penalizacion_z= (zsup- z)**2
    return penalizacion_z


#definir la funcion error conjunto
def error_conjunto_penalizado(variables):
    Vo, R1, R2, R3, L = variables
    K1= 10
    K2= 10
    K3= 1e6
    K4= 1e6
    K5= 1e20
    K6= 1e6
    K7 = 1e8
    error_v = np.sum((tensionca(t_data, Vo, R1, R2, R3, L)- refu_t_np(t_data))**2)
    error_i = np.sum((corrcc_no_negatives(t_data, Vo, R1, R2, R3, L)- refi_t_np(t_data))**2)
    error_tfu= penalizacion_tfu([Vo, R1, R2, R3, L])
    error_tcu= penalizacion_tc_u([Vo, R1, R2, R3, L])
    error_tfi= penalizacion_tfi([Vo, R1, R2, R3, L])
    error_tci= penalizacion_tc_i([Vo, R1, R2, R3, L])
    error_z= penalizacion_z([Vo, R1, R2, R3, L])
    return K1 * error_v +  K2 * error_i + K3 * error_tfu + K4 * error_tfi + K5 * error_z + K6 * error_tcu + K7 * error_tci



def optimizar_parametros(C_value):
    global C
    C = C_value
    
    # Generar datos de tiempo
    t_data = np.linspace(1e-7, 100e-6, num=6000)

    # Evaluar las funciones en los datos de tiempo
    refu_t_eval = refu_t_np(t_data)
    refi_t_eval = refi_t_np(t_data)

    # Valores iniciales para los parámetros
    parametros_iniciales = [Vox, R1x, R2x, R3x, Lx]

    resultado = minimize(error_conjunto_penalizado, parametros_iniciales, method='Nelder-Mead')

    # Obtener los parámetros óptimos del resultado de la optimización
    Vo_opt, R1_opt, R2_opt, R3_opt, L_opt = resultado.x

    # Calcular el error mínimo
    error_minimo = resultado.fun

    # Devolver los resultados
    return (Vo_opt, R1_opt, R2_opt, R3_opt, L_opt), error_minimo

# Lista para almacenar los resultados
resultados = []
matriz_parametros= []
valores_zp=[]
valores_tfu=[]
valores_tcu=[]
valores_tfi=[]
valores_tci=[]
Capacitancia=[]
valores_sobrep=[]

# Valores de capacitancia a probar
#C_values = np.arange(3e-6, 3.5e-6, 0.5e-6)

for numero in range(50, 150):
    C = numero*1e-6/10
    # Generar datos de tiempo
    t_data = np.linspace(1e-7, 100e-6, num=6000)
    # Evaluar las funciones en los datos de tiempo
    refu_t_eval = refu_t_np(t_data)
    refi_t_eval = refi_t_np(t_data)
    # Valores iniciales para los parámetros
    parametros_iniciales = [Vox, R1x, R2x, R3x, Lx]
    resultado = minimize(error_conjunto_penalizado, parametros_iniciales, method='Nelder-Mead')
    # Obtener los parámetros óptimos del resultado de la optimización
    Vo_opt, R1_opt, R2_opt, R3_opt, L_opt = resultado.x
    # Calcular el error mínimo
    error_minimo = resultado.fun
    print(f"Para C = {C}:")
    print("Parámetros óptimos:", resultado.x)
    print("Error mínimo:", error_minimo)
    valor_zp = zp(resultado.x)
    valores_zp.append(valor_zp)
    print("zp:", valor_zp)
    valor_tfu = tf_u(resultado.x)
    valores_tfu.append(valor_tfu)
    print("tfu:", valor_tfu)
    valor_tcu = tc_u(resultado.x)
    valores_tcu.append(valor_tcu)
    print("tcu:", valor_tcu)
    valor_tfi = tf_i(resultado.x)
    valores_tfi.append(valor_tfi)
    print("tfi:", valor_tfi)
    valor_tci = tc_i(resultado.x)
    valores_tci.append(valor_tci)
    print("tci:", valor_tci)
    #Cálculo valor del sobrepaso negativo de la corriente
    imax = max(corrcc(t_data, Vo_opt, R1_opt, R2_opt, R3_opt, L_opt))
    imin = min(corrcc(t_data, Vo_opt, R1_opt, R2_opt, R3_opt, L_opt))
    sobrepaso_i=imin/imax
    valores_sobrep.append(sobrepaso_i)
    print("Sobrepaso - Corriente:", sobrepaso_i)
    print()
    Capacitancia.append(C)
    resultados.append(error_minimo)
    matriz_parametros.append(resultado.x)


# Crear un DataFrame con los resultados
df_resultados = pd.DataFrame({
    'Capacitancia': Capacitancia,
    'Error Mínimo': resultados,
    'Valor zp': valores_zp,
    'Valor tfu': valores_tfu,
    'Valor tcu': valores_tcu,
    'Valor tfi': valores_tfi,
    'Valor tci': valores_tci,
    'Sobrepaso Corriente': valores_sobrep
})

df_parametros = pd.DataFrame(matriz_parametros, columns=['Vo', 'R1', 'R2', 'R3', 'L'])
df_resultados = pd.concat([df_resultados, df_parametros], axis=1)

# Guardar el DataFrame en un archivo Excel
nombre_excel = "resultados2.xlsx"
df_resultados.to_excel(nombre_excel, index=False)

print(f"Se han guardado los resultados en {nombre_excel}")


# # Graficar los errores mínimos en función de los valores de C
plt.figure(figsize=(10, 6))
plt.plot(Capacitancia, resultados, marker='o', linestyle='-')
plt.xlabel('Valor de C')
plt.ylabel('Error mínimo')
plt.title('Relación entre el valor de C y el error mínimo')
plt.grid(True)
plt.show()

# # Graficar los sobrepasos (pu) de la corriente en función de los valores de C
plt.figure(figsize=(10, 6))
plt.plot(Capacitancia, valores_sobrep, marker='o', linestyle='-', label='Valores de Sobrepaso')
plt.axhline(y=-0.3, color='red', linestyle='-', label='S-max 30%')
plt.xlabel('Valor de C')
plt.ylabel('Sobrepaso de la Corriente en (PU)')
plt.title('Relación entre el valor de C y el sobrepaso')
plt.grid(True)
plt.show()

# # Graficar los valores de zp en función de los valores de C
plt.figure(figsize=(10, 6))
plt.plot(Capacitancia, valores_zp, marker='o', linestyle='-', label='Valores de zp')
plt.axhline(y=2, color='black', linestyle='-', label='Valor constante 2')
plt.axhline(y=2.44, color='red', linestyle='--', label='Valor constante 2.44', alpha=0.7)
plt.axhline(y=1.63, color='red', linestyle='--', label='Valor constante 1.63', alpha=0.7)
plt.xlabel('Valor de C')
plt.ylabel('Valor de zp')
plt.title('Relación entre el valor de C y el valor de zp')
plt.grid(True)
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(Capacitancia, valores_tfi, marker='o', linestyle='-', label='Valores de tf_i')
plt.axhline(y=8e-6, color='black', linestyle='-', label='Valor constante 8e-6')
plt.axhline(y=6.4e-6, color='red', linestyle='--', label='Valor constante 6.4e-6', alpha=0.7)
plt.axhline(y=9.6e-6, color='red', linestyle='--', label='Valor constante 9.6e-6', alpha=0.7)
plt.xlabel('Valor de C')
plt.ylabel('Valor de tf_i')
plt.title('Relación entre el valor de C y el valor de tf_i')
plt.grid(True)
plt.legend()
plt.show()

# # Graficar los valores de tc_i en función de los valores de C
plt.figure(figsize=(10, 6))
plt.plot(Capacitancia, valores_tci, marker='o', linestyle='-', label='Valores de tc_i')
plt.axhline(y=20e-6, color='black', linestyle='-', label='Valor constante 20e-6')
plt.axhline(y=16e-6, color='red', linestyle='--', label='Valor constante 16e-6', alpha=0.7)
plt.axhline(y=24e-6, color='red', linestyle='--', label='Valor constante 24e-6', alpha=0.7)
plt.xlabel('Valor de C')
plt.ylabel('Valor de tc_i')
plt.title('Relación entre el valor de C y el valor de tc_i')
plt.grid(True)
plt.legend()
plt.show()

# # Graficar los valores de tf_u en función de los valores de C
plt.figure(figsize=(10, 6))
plt.plot(Capacitancia, valores_tfu, marker='o', linestyle='-', label='Valores de tf_u')
plt.axhline(y=1.2e-6, color='black', linestyle='-', label='Valor constante 1.2e-6')
plt.axhline(y=0.87e-6, color='red', linestyle='--', label='Valor constante 0.87e-6', alpha=0.7)
plt.axhline(y=1.56e-6, color='red', linestyle='--', label='Valor constante 1.56e-6', alpha=0.7)
plt.xlabel('Valor de C')
plt.ylabel('Valor de tf_u')
plt.title('Relación entre el valor de C y el valor de tf_u')
plt.grid(True)
plt.legend()
plt.show()

# # Graficar los valores de tc_u en función de los valores de C
plt.figure(figsize=(10, 6))
plt.plot(Capacitancia, valores_tcu, marker='o', linestyle='-', label='Valores de tc_u')
plt.axhline(y=50e-6, color='black', linestyle='-', label='Valor constante 50e-6')
plt.axhline(y=40e-6, color='red', linestyle='--', label='Valor constante 40e-6', alpha=0.7)
plt.axhline(y=60e-6, color='red', linestyle='--', label='Valor constante 60e-6', alpha=0.7)
plt.xlabel('Valor de C')
plt.ylabel('Valor de tc_u')
plt.title('Relación entre el valor de C y el valor de tc_u')
plt.grid(True)
plt.legend()
plt.show()



input("Presiona Enter para salir...")