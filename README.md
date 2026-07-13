# Detecção automática de flakes por microscopia

**Varredura automatizada de amostras
em microscópio motorizado** e **detecção de flakes**com o
modelo [MaskTerial](https://github.com/Jaluus/MaskTerial).

O microscópio é controlado via [pycromanager](https://pycro-manager.readthedocs.io/).

---

## Estrutura dos arquivos

### Configuração (edite aqui os valores)
| Arquivo | Função |
|---|---|
| `caminhos.py` | Caminhos das pastas: pesos do modelo e diretório de saída. |
| `parametros.py` | Parâmetros de aquisição e detecção (FOV, exposição, balanço de branco, limiares). Preencha com o que sai das calibrações. |

### Módulos compartilhados (não se rodam sozinhos)
| Arquivo | Função |
|---|---|
| `comandos.py` | Cálculos comuns: ajuste do plano/superfície de foco, geração da malha de varredura e coleta de pontos de calibração. |
| `modelo.py` | Carrega o MaskTerial (uma vez, sob demanda) e desenha os flakes detectados sobre a imagem. |
| `capturar_imagem.py` | Captura uma imagem da câmera e devolve em RGB. |

### Scripts que se rodam
| Arquivo | Função |
|---|---|
| `propriedades_camera.py` | Lista as propriedades de cor da câmera (útil para descobrir os nomes de `WhiteBalance...`). |
| `calibrar_camera.py` | Calibra balanço de branco e exposição. Copie os valores para `parametros.py`. |
| `calibrar_fov.py` | Mede o campo de visão (µm) por correlação de fase. Copie o valor para `parametros.py`. |
| `deteccao.py` | **Pipeline principal**: varre a amostra, ajusta o foco (com autofoco periódico), detecta flakes e salva imagens + um CSV com as coordenadas. |
| `varredura_mosaico.py` | Varredura de visão geral (objetiva de menor magnificação): fotografa a grade e monta um mosaico único. Não detecta flakes. |
| `testar_modelo_pasta.py` | Roda o modelo em uma pasta de imagens `.tif` **sem microscópio**, para conferir a detecção. |

----
## Fluxo de uso

```
1. propriedades_camera.py   (opcional) descobrir nomes das propriedades de cor
2. calibrar_camera.py       balanço de branco + exposição  ─┐
3. calibrar_fov.py          campo de visão (µm)            ─┴─► copiar para parametros.py
4. deteccao.py              varredura + detecção
   (ou varredura_mosaico.py para uma visão geral em mosaico)
```

Os resultados de `deteccao.py` vão para uma pasta `Varredura_<data-hora>` dentro
de `DIRETORIO_SAIDA_BASE` (`caminhos.py`), com as imagens dos flakes, cópias de
conferência e o CSV `coordenadas_flakes_*.csv`.

---

## Observações
- **Onde mexer**: valores de calibração ficam em `parametros.py`; caminhos em
  `caminhos.py`. Evite espalhar números soltos pelos scripts.
