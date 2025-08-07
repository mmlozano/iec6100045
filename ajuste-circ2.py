import numpy as np
import sympy as sp
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.optimize import minimize
from scipy.optimize import fsolve

# Definición de los parámetros
Vox = 1
R1x = 50
Rm1x = 15
Rm2x= 25
C2x = 0.22e-6
C1 = 20e-6
Zopt=40

# Definir el tiempo simbólico
t = sp.symbols('t')


# Definir la función tension-ca(t)
def tensionca(x, Vot, R1t, Rm1t, Rm2t, C2t):
    expr = 2*C1*R1t*Vot*sp.exp(-t*(C1*R1t + C2t*R1t + C2t*Rm1t)/(2*C1*C2t*R1t*Rm1t))*sp.sin(t*sp.sqrt(-C1**2*R1t**2 - 2*C1*C2t*R1t**2 + 2*C1*C2t*R1t*Rm1t - C2t**2*R1t**2 - 2*C2t**2*R1t*Rm1t - C2t**2*Rm1t**2)/(2*C1*C2t*R1t*Rm1t))/sp.sqrt(-C1**2*R1t**2 - 2*C1*C2t*R1t**2 + 2*C1*C2t*R1t*Rm1t - C2t**2*R1t**2 - 2*C2t**2*R1t*Rm1t - C2t**2*Rm1t**2)
        
    #expr = 2*Vot*sp.exp(-t*(1/(C2t*Rm1t) + 1/(C1*Rm1t) + 1/(C1*R1t))/2)* \
    #sp.sin(t*sp.sqrt(-1/(C2t**2*Rm1t**2) - 2/(C1*C2t*Rm1t**2) + 2/(C1*C2t*R1t*Rm1t) - 1/(C1**2*Rm1t**2) - 2/(C1**2*R1t*Rm1t) - 1/(C1**2*R1t**2))/2)* \
    #sp.Heaviside(t)/(C2t*Rm1t*sp.sqrt(-1/(C2t**2*Rm1t**2) - 2/(C1*C2t*Rm1t**2) + 2/(C1*C2t*R1t*Rm1t) - 1/(C1**2*Rm1t**2) - 2/(C1**2*R1t*Rm1t) - 1/(C1**2*R1t**2)))
    
    expr2 = sp.lambdify(t, expr, modules=['numpy'])
    return expr2(x)

# Definir la función corriente-cc(t)
def corrcc(x, Vot, R1t, Rm1t, Rm2t, C2t):
    exprc= 2*C1*R1t*Vot*sp.exp(-t*(C1*R1t*Rm1t + C1*R1t*Rm2t + C2t*R1t*Rm2t + C2t*Rm1t*Rm2t)/(2*C1*C2t*R1t*Rm1t*Rm2t))*sp.sin(t*sp.sqrt(-C1**2*R1t**2*Rm1t**2 - 2*C1**2*R1t**2*Rm1t*Rm2t - 
    C1**2*R1t**2*Rm2t**2 + 2*C1*C2t*R1t**2*Rm1t*Rm2t - 2*C1*C2t*R1t**2*Rm2t**2 + 2*C1*C2t*R1t*Rm1t**2*Rm2t + 2*C1*C2t*R1t*Rm1t*Rm2t**2 - C2t**2*R1t**2*Rm2t**2 - 2*C2t**2*R1t*Rm1t*Rm2t**2 - C2t**2*Rm1t**2*Rm2t**2)/(2*C1*C2t*R1t*Rm1t*Rm2t))/sp.sqrt(-C1**2*R1t**2*Rm1t**2 - 2*C1**2*R1t**2*Rm1t*Rm2t - C1**2*R1t**2*Rm2t**2 + 2*C1*C2t*R1t**2*Rm1t*Rm2t - 2*C1*C2t*R1t**2*Rm2t**2 + 2*C1*C2t*R1t*Rm1t**2*Rm2t + 2*C1*C2t*R1t*Rm1t*Rm2t**2 - C2t**2*R1t**2*Rm2t**2 - 2*C2t**2*R1t*Rm1t*Rm2t**2 - C2t**2*Rm1t**2*Rm2t**2) 
  
   # exprc = 2*Vot*sp.exp(-t*(1/(C2t*Rm2t) + 1/(C2t*Rm1t) + 1/(C1*Rm1t) + 1/(C1*R1t))/2)* \
   # sp.sin(t*sp.sqrt(-1/(C2t**2*Rm2t**2) - 2/(C2t**2*Rm1t*Rm2t) - 1/(C2t**2*Rm1t**2) + 2/(C1*C2t*Rm1t*Rm2t) - 2/(C1*C2t*Rm1t**2) + 2/(C1*C2t*R1t*Rm2t) + 2/(C1*C2t*R1t*Rm1t) - 1/(C1**2*Rm1t**2) - 2/(C1**2*R1t*Rm1t) - 1/(C1**2*R1t**2))/2)* \
   # sp.Heaviside(t)/(C2t*Rm1t*Rm2t*sp.sqrt(-1/(C2t**2*Rm2t**2) - 2/(C2t**2*Rm1t*Rm2t) - 1/(C2t**2*Rm1t**2) + 2/(C1*C2t*Rm1t*Rm2t) - 2/(C1*C2t*Rm1t**2) + 2/(C1*C2t*R1t*Rm2t) + 2/(C1*C2t*R1t*Rm1t) - 1/(C1**2*Rm1t**2) - 2/(C1**2*R1t*Rm1t) - 1/(C1**2*R1t**2)))
    exprc2 = sp.lambdify(t, exprc, modules=['numpy'])
    return exprc2(x)


# Definir la función refu(t)
def refu_t(t):
    tau1 = 2.574e-6  # segundos
    tau2 = 945.1e-6  # segundos
    v1 = 0.937
    eta = 1.749
    kV = 1
    k_onda= np.exp( - (tau1 / tau2) * ((eta * tau2 / tau1) ** (1 / eta)) )
    base = (t / tau1) ** eta
    return kV * (v1 / k_onda) * (base / (1 + base)) * np.exp(-t / tau2)


# Definir la función refi(t)
def refi_t(t):
    tau1 = 1.355e-6  # s
    tau2 = 429.1e-6  # s
    eta = 1.556
    i1 = 0.895
    k_i = 1
    k_onda = np.exp(- (tau1 / tau2) * eta)
    base = (t / tau1) ** eta
    return 0.25*k_i * (i1 / k_onda) * (base / (1 + base)) * np.exp(-t / tau2)



# Generar datos de tiempo
t_data = np.linspace(1e-7, 1400e-6, num=6000)

# Evaluar las funciones en los datos de tiempo
refu_t_eval = refu_t(t_data)
refi_t_eval = refi_t(t_data)
# Y las eñales evaluadas en los valores iniciales de la iteracion
tensionca_t_eval = tensionca(t_data, Vox, R1x, Rm1x, Rm2x, C2x)
corrcc_t_eval = corrcc(t_data, Vox, R1x, Rm1x, Rm2x, C2x)

# Graficar antes de la optimización
plt.ion()
plt.figure(figsize=(10, 6))
plt.plot(t_data, tensionca_t_eval, label='u_eval(t)')
plt.plot(t_data, corrcc_t_eval, label='i_eval(t)')
plt.plot(t_data, refu_t_eval, label='ref_u(t)')
plt.plot(t_data, refi_t_eval, label='ref_i(t)')
plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud')
plt.legend()
plt.grid(True)
plt.title('Antes de la optimización')
plt.show()


def tf_u(variables):
    Vo, R1, Rm1, Rm2, C2= variables
    umax=max(tensionca(t_data, Vo, R1, Rm1, Rm2, C2))
    indice_t_30= np.argmax(tensionca(t_data, Vo, R1, Rm1, Rm2, C2) >= (0.3 * umax))
    t_u_30=t_data[indice_t_30]
    indice_t_90= np.argmax(tensionca(t_data, Vo, R1, Rm1, Rm2, C2) >= (0.9 * umax))
    t_u_90=t_data[indice_t_90] 
    return 1.67 * (t_u_90 - t_u_30)   

def tc_u(variables):
    Vo, R1, Rm1, Rm2, C2= variables
    umax = max(tensionca(t_data, Vo, R1, Rm1, Rm2, C2))
    #print("umax: ", umax)
    indice_t_501 = np.argmax(tensionca(t_data, Vo, R1, Rm1, Rm2, C2) >= (0.5 * umax))
    t_u_501 = t_data[indice_t_501]
    #print("t_501: ", t_u_501)
    u501 = tensionca(t_u_501, Vo, R1, Rm1, Rm2, C2)
    #print("u_501: ", u501)
    t_data2 = np.linspace(t_u_501+10e-6, 1000e-6, num=6000)
    indice_t_502 = np.argmax(tensionca(t_data2, Vo, R1, Rm1, Rm2, C2) <= (0.5 * umax))
    t_u_502 = t_data2[indice_t_502] 
    #print("t_u_502", t_u_502)
    u502 = tensionca(t_u_502, Vo, R1, Rm1, Rm2, C2)
    #print("u_502: ", u502)
    return (t_u_502-t_u_501)

def tf_i(variables):
    Vo, R1, Rm1, Rm2, C2 = variables
    imax=max(corrcc(t_data, Vo, R1, Rm1, Rm2, C2))
    indice_t_10= np.argmax(corrcc(t_data, Vo, R1, Rm1, Rm2, C2) >= (0.1 * imax))
    t_i_10=t_data[indice_t_10]
    indice_t_90= np.argmax(corrcc(t_data, Vo, R1, Rm1, Rm2, C2) >= (0.9 * imax))
    t_i_90=t_data[indice_t_90] 
    return 1.25 * (t_i_90 - t_i_10)   


def tc_i(variables):
    Vo, R1, Rm1, Rm2, C2= variables
    imax = max(corrcc(t_data, Vo, R1, Rm1, Rm2, C2))
    #print("imax: ", imax)
    indice_t_501 = np.argmax(corrcc(t_data, Vo, R1, Rm1, Rm2, C2) >= (0.5 * imax))
    t_i_501 = t_data[indice_t_501]
    #print("t_501: ", t_i_501)
    i501 = corrcc(t_i_501, Vo, R1, Rm1, Rm2, C2)
    #print("i_501: ", i501)
    t_data2 = np.linspace(t_i_501+10e-6, 1000e-6, num=6000)
    indice_t_502 = np.argmax(corrcc(t_data2, Vo, R1, Rm1, Rm2, C2) <= (0.5 * imax))
    t_i_502 = t_data2[indice_t_502] 
    #print("t_i_502", t_i_502)
    i502 = corrcc(t_i_502, Vo, R1, Rm1, Rm2, C2)
    #print("i_502: ", i502)
    return 1.18*(t_i_502-t_i_501)

def zp(variables):
    Vo, R1, Rm1, Rm2, C2 = variables
    Zp= max(tensionca(t_data, Vo, R1, Rm1, Rm2, C2))/max(corrcc(t_data, Vo, R1, Rm1, Rm2, C2))
    return 1 * Zp


def penalizacion_tfu(variables):
    Vo, R1, Rm1, Rm2, C2= variables
    tinf=7e-6
    tsup=13e-6
    tn=10e-6
    tiempo=tf_u([Vo, R1, Rm1, Rm2, C2])
    penalizacion_u = 0.1*(tn*1e6- tiempo*1e6)**2
    if tiempo < tinf:
        penalizacion_u = 1000+1000*(tinf*1e6- tiempo*1e6)**2
    elif tiempo > tsup:
        penalizacion_u = 1000+1000*(tsup*1e6- tiempo*1e6)**2

    return penalizacion_u

def penalizacion_tfi(variables):
    Vo, R1, Rm1, Rm2, C2 = variables
    tinf=4e-6
    tsup=6e-6
    tn=5e-6
    tiempo=tf_i([Vo, R1, Rm1, Rm2, C2])
    penalizacion_i = 0.1*(tn*1e6- tiempo*1e6)**2
    if tiempo < tinf:
        penalizacion_i =1000+1000*(tinf*1e6- tiempo*1e6)**2
    elif tiempo > tsup:
        penalizacion_i= 1000+1000*(tsup*1e6- tiempo*1e6)**2

    return penalizacion_i


def penalizacion_tc_u(variables):
    Vo, R1, Rm1, Rm2, C2 = variables
    tinf=560e-6
    tsup=840e-6
    tn=700e-6
    tiempo=tc_u([Vo, R1, Rm1, Rm2, C2])
    penalizacion_tcu = 0.1*(tn*1e6- tiempo*1e6)**2
    if tiempo < tinf:
        penalizacion_tcu = 1000+1000*(tinf*1e6- tiempo*1e6)**2
    elif tiempo > tsup:
        penalizacion_u = 1000+1000*(tsup*1e6- tiempo*1e6)**2
    return penalizacion_tcu


def penalizacion_tc_i(variables):
    Vo, R1, Rm1, Rm2, C2 = variables
    tinf=256e-6
    tsup=384e-6
    tn=320e-6
    tiempo=tc_i([Vo, R1, Rm1, Rm2, C2])
    penalizacion_tci = 0.1*(tn*1e6- tiempo*1e6)**2
    if tiempo < tinf:
        penalizacion_tci = 1000+1000*(tinf*1e6- tiempo*1e6)**2
    elif tiempo > tsup:
        penalizacion_tci = 1000+1000*(tsup*1e6- tiempo*1e6)**2
    return penalizacion_tci



def penalizacion_z(variables):
    Vo, R1, Rm1, Rm2, C2= variables
    zinf=32.73
    zsup=48.80
    zn=40
    z=zp([Vo, R1, Rm1, Rm2, C2])
    penalizacion_z = 0.1*(zn- z)**2
    if z < zinf:
        penalizacion_z = 1000+1000*(zinf- z)**2
    elif z > zsup:
        penalizacion_z= 1000+1000*(zsup- z)**2
    return penalizacion_z


#definir la funcion error conjunto
def error_conjunto_penalizado(variables):
    Vo, R1, Rm1, Rm2, C2 = variables
    K3= 100
    K4= 100
    K5= 100
    K6= 100
    K7 = 100
    error_tfu= penalizacion_tfu([Vo, R1, Rm1, Rm2, C2])
    error_tcu= penalizacion_tc_u([Vo, R1, Rm1, Rm2, C2])
    error_tfi= penalizacion_tfi([Vo, R1, Rm1, Rm2, C2])
    error_tci= penalizacion_tc_i([Vo, R1, Rm1, Rm2, C2])
    error_z= penalizacion_z([Vo, R1, Rm1, Rm2, C2])
    return K3 * error_tfu + K4 * error_tfi + K5 * error_z + K6 * error_tcu + K7 * error_tci



def optimizar_parametros(C_value):
    global C
    C1 = C_value
    
    # Generar datos de tiempo
    t_data = np.linspace(1e-7, 1400e-6, num=6000)

    # Evaluar las funciones en los datos de tiempo
    refu_t_eval = refu_t(t_data)
    refi_t_eval = refi_t(t_data)

    # Valores iniciales para los parámetros
    parametros_iniciales = [Vox, R1x, Rm1x, Rm2x, C2x]

    resultado = minimize(error_conjunto_penalizado, parametros_iniciales, method='Nelder-Mead')

    # Obtener los parámetros óptimos del resultado de la optimización
    Vo_opt, R1_opt, Rm1_opt, Rm2opt, C2_opt = resultado.x

    # Calcular el error mínimo
    error_minimo = resultado.fun

    # Devolver los resultados
    return (Vo_opt, R1_opt, Rm1_opt, Rm2opt, C2_opt), error_minimo

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

for numero in range(50, 300, 5):
    C1 = numero*1e-7
    # Generar datos de tiempo
    t_data = np.linspace(1e-7, 1000e-6, num=10000)
    # Evaluar las funciones en los datos de tiempo
    refu_t_eval = refu_t(t_data)
    refi_t_eval = refi_t(t_data)
    # Valores iniciales para los parámetros
    parametros_iniciales = [Vox, R1x, Rm1x, Rm2x, C2x]
    resultado = minimize(error_conjunto_penalizado, parametros_iniciales, method='Nelder-Mead')
    # Obtener los parámetros óptimos del resultado de la optimización
    Vo_opt, R1_opt, Rm1_opt, Rm2opt, C2_opt = resultado.x
    # Calcular el error mínimo
    error_minimo = resultado.fun
    print(f"Para C = {C1}:")
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
    imax = max(corrcc(t_data, Vo_opt, R1_opt, Rm1_opt, Rm2opt, C2_opt))
    imin = min(corrcc(t_data, Vo_opt, R1_opt, Rm1_opt, Rm2opt, C2_opt))
    sobrepaso_i=imin/imax
    valores_sobrep.append(sobrepaso_i)
    print("Sobrepaso - Corriente:", sobrepaso_i)
    print()
    Capacitancia.append(C1)
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

df_parametros = pd.DataFrame(matriz_parametros, columns=['Vo', 'R1', 'Rm1', 'Rm2', 'C2'])
df_resultados = pd.concat([df_resultados, df_parametros], axis=1)

# Guardar el DataFrame en un archivo Excel
nombre_excel = "resultados2.xlsx"
df_resultados.to_excel(nombre_excel, index=False)

print(f"Se han guardado los resultados en {nombre_excel}")


# # Graficar los errores mínimos en función de los valores de C
plt.figure(figsize=(10, 6))
plt.plot(Capacitancia, resultados, marker='o', linestyle='-')
plt.xlabel('Valor de C1')
plt.ylabel('Error mínimo')
plt.title('Relación entre el valor de C y el error mínimo')
plt.grid(True)
plt.show()

# # Graficar los sobrepasos (pu) de la corriente en función de los valores de C
plt.figure(figsize=(10, 6))
plt.plot(Capacitancia, valores_sobrep, marker='o', linestyle='-', label='Valores de Sobrepaso')
plt.axhline(y=-0.3, color='red', linestyle='-', label='S-max 30%')
plt.xlabel('Valor de C1')
plt.ylabel('Sobrepaso de la Corriente en (PU)')
plt.title('Relación entre el valor de C1 y el sobrepaso')
plt.grid(True)
plt.show()

# # Graficar los valores de zp en función de los valores de C
plt.figure(figsize=(10, 6))
plt.plot(Capacitancia, valores_zp, marker='o', linestyle='-', label='Valores de zp')
plt.axhline(y=40, color='black', linestyle='-', label='Valor constante 2')
plt.axhline(y=48.8888, color='red', linestyle='--', label='Valor constante 2.44', alpha=0.7)
plt.axhline(y=32.7272, color='red', linestyle='--', label='Valor constante 1.63', alpha=0.7)
plt.xlabel('Valor de C1')
plt.ylabel('Valor de zp')
plt.title('Relación entre el valor de C1 y el valor de zp')
plt.grid(True)
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(Capacitancia, valores_tfi, marker='o', linestyle='-', label='Valores de tf_i')
plt.axhline(y=5e-6, color='black', linestyle='-', label='Valor constante 8e-6')
plt.axhline(y=4e-6, color='red', linestyle='--', label='Valor constante 6.4e-6', alpha=0.7)
plt.axhline(y=6e-6, color='red', linestyle='--', label='Valor constante 9.6e-6', alpha=0.7)
plt.xlabel('Valor de C1')
plt.ylabel('Valor de tf_i')
plt.title('Relación entre el valor de C y el valor de tf_i')
plt.grid(True)
plt.legend()
plt.show()

# # Graficar los valores de tc_i en función de los valores de C
plt.figure(figsize=(10, 6))
plt.plot(Capacitancia, valores_tci, marker='o', linestyle='-', label='Valores de tc_i')
plt.axhline(y=320e-6, color='black', linestyle='-', label='Valor constante 20e-6')
plt.axhline(y=256e-6, color='red', linestyle='--', label='Valor constante 16e-6', alpha=0.7)
plt.axhline(y=384e-6, color='red', linestyle='--', label='Valor constante 24e-6', alpha=0.7)
plt.xlabel('Valor de C1')
plt.ylabel('Valor de tc_i')
plt.title('Relación entre el valor de C y el valor de tc_i')
plt.grid(True)
plt.legend()
plt.show()

# # Graficar los valores de tf_u en función de los valores de C
plt.figure(figsize=(10, 6))
plt.plot(Capacitancia, valores_tfu, marker='o', linestyle='-', label='Valores de tf_u')
plt.axhline(y=10e-6, color='black', linestyle='-', label='Valor constante 1.2e-6')
plt.axhline(y=7e-6, color='red', linestyle='--', label='Valor constante 0.87e-6', alpha=0.7)
plt.axhline(y=13e-6, color='red', linestyle='--', label='Valor constante 1.56e-6', alpha=0.7)
plt.xlabel('Valor de C1')
plt.ylabel('Valor de tf_u')
plt.title('Relación entre el valor de C y el valor de tf_u')
plt.grid(True)
plt.legend()
plt.show()

# # Graficar los valores de tc_u en función de los valores de C
plt.figure(figsize=(10, 6))
plt.plot(Capacitancia, valores_tcu, marker='o', linestyle='-', label='Valores de tc_u')
plt.axhline(y=700e-6, color='black', linestyle='-', label='Valor constante 50e-6')
plt.axhline(y=560e-6, color='red', linestyle='--', label='Valor constante 40e-6', alpha=0.7)
plt.axhline(y=840e-6, color='red', linestyle='--', label='Valor constante 60e-6', alpha=0.7)
plt.xlabel('Valor de C1')
plt.ylabel('Valor de tc_u')
plt.title('Relación entre el valor de C y el valor de tc_u')
plt.grid(True)
plt.legend()
plt.show()



input("Presiona Enter para salir...")