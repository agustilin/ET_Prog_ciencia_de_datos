CREATE TABLE perfil_usuarios
(
    id_cliente INT PRIMARY KEY,
    edad INT,
    dispositivos_registrados INT,
    porcentaje_uso_app_movil NUMERIC,
    cantidad_perfiles_creados INT,
    interacciones_mensuales_soporte INT,
    distancia_promedio_red_km NUMERIC
);

COPY perfil_usuarios
FROM '/docker-entrypoint-initdb.d/perfil_usuarios.csv'
DELIMITER ','
CSV HEADER;