from pycromanager import Core

core = Core()
camera = core.get_camera_device()

print("Câmera ativa:", camera)


def str_vector_para_lista(vetor):
    """
    Converte objetos tipo mmcorej_StrVector em lista Python.
    Também funciona se o objeto já for uma lista/tupla.
    """

    # Caso já seja iterável como lista Python
    try:
        return list(vetor)
    except TypeError:
        pass

    # Caso seja mmcorej_StrVector
    lista = []
    for i in range(vetor.size()):
        lista.append(vetor.get(i))

    return lista


props = str_vector_para_lista(core.get_device_property_names(camera))

palavras_chave = [
    "white",
    "balance",
    "red",
    "green",
    "blue",
    "color",
    "gain",
    "gamma",
    "pixel",
    "rgb",
    "auto",
    "temperature",
]

print("\n--- Propriedades possivelmente relacionadas a cor ---")

for prop in props:
    prop_lower = prop.lower()

    if any(palavra in prop_lower for palavra in palavras_chave):
        print(f"\nPropriedade: {prop}")

        try:
            valor = core.get_property(camera, prop)
            print(f"  valor atual: {valor}")
        except Exception as erro:
            print(f"  não consegui ler valor atual: {erro}")

        try:
            read_only = core.is_property_read_only(camera, prop)
            print(f"  read-only: {read_only}")
        except Exception as erro:
            print(f"  não consegui verificar se é read-only: {erro}")

        try:
            permitido = core.get_allowed_property_values(camera, prop)
            permitido = str_vector_para_lista(permitido)

            if len(permitido) > 0:
                print(f"  valores permitidos: {permitido}")
        except Exception as erro:
            print(f"  não consegui ler valores permitidos: {erro}")

        try:
            if core.has_property_limits(camera, prop):
                minimo = core.get_property_lower_limit(camera, prop)
                maximo = core.get_property_upper_limit(camera, prop)
                print(f"  limites: {minimo} até {maximo}")
        except Exception as erro:
            print(f"  não consegui ler limites: {erro}")