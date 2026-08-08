# Detecção automática de flakes por microscopia

**Varredura automatizada de amostras em microscópio motorizado** e **detecção de
flakes** com o modelo [MaskTerial](https://github.com/Jaluus/MaskTerial).

O microscópio é controlado via [pycromanager](https://pycro-manager.readthedocs.io/).

---

## Instalação

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

**O `detectron2` exige atenção manual.** Ele compila extensões nativas (precisa
de um compilador C++), não tem wheel no PyPI e precisa do `torch` já instalado
antes de compilar. Se o `pip install -r requirements.txt` falhar nele, instale o
`torch` sozinho primeiro e repita. Ver `NOTAS_REVISAO.md`.

Os pesos do modelo não vêm no repositório: aponte `SEG_MODEL_ROOT` e
`CLS_MODEL_ROOT` em `caminhos.py` para onde eles estiverem.

---

## Estrutura dos arquivos

### Configuração (é aqui que se edita)

| Arquivo | Função |
|---|---|
| `caminhos.py` | Caminhos das pastas: pesos do modelo e diretório de saída. **Diverge de propósito entre máquinas** — não sincronize. |
| `parametros.py` | Todos os parâmetros de aquisição e detecção: FOV, exposição, balanço de branco, sobreposição, limiares do MaskTerial e as bandas de escala da métrica de foco. Preencha com o que sai das calibrações. |

### Módulos compartilhados (não se rodam sozinhos)

| Arquivo | Função |
|---|---|
| `comandos.py` | Núcleo do projeto. Ajuste do plano/superfície de foco, sugestão e coleta guiada dos pontos de calibração, geração da malha de varredura, métricas de imagem e autofoco. |
| `modelo.py` | Carrega o MaskTerial (uma vez, sob demanda) e desenha os flakes detectados sobre a imagem. |
| `capturar_imagem.py` | Captura uma imagem da câmera e devolve em RGB. |

### Aquisição

| Arquivo | Função |
|---|---|
| `deteccao.py` | **Pipeline principal.** Varre a amostra, ajusta o foco (modelo + autofoco periódico que realimenta o modelo), detecta flakes e salva imagens + CSVs. |
| `varredura_mosaico.py` | Varredura de visão geral, com a objetiva de menor magnificação. Fotografa a grade; **não** detecta flakes. |

### Pós-processamento

| Arquivo | Função |
|---|---|
| `montar_mosaico.py` | Monta o mosaico final a partir das imagens que uma varredura deixou em disco. Roda offline, sem microscópio, e é repetível. |

### Calibração

| Arquivo | Função |
|---|---|
| `propriedades_camera.py` | Lista as propriedades de cor da câmera (para descobrir os nomes de `WhiteBalance...`). |
| `calibrar_camera.py` | Balanço de branco e exposição. Copie os valores para `parametros.py`. |
| `calibrar_fov.py` | Mede o campo de visão (µm) por correlação de fase. Copie o valor para `parametros.py`. |
| `calibrar_razao_foco.py` | Levanta a curva da razão de foco contra Z (z-stack). É daqui que sai o limiar da métrica de foco. Salva o stack em disco para reanálise offline. |

### Conferência e testes (sem microscópio)

| Arquivo | Função |
|---|---|
| `testar_calibracao_foco.py` | Testes de regressão da calibração de foco. Roda em qualquer máquina; sai com código 1 se algo falhar. |
| `testar_modelo_pasta.py` | Roda o modelo numa pasta de imagens `.tif`, para conferir a detecção sem varrer. |
| `analisar_foco_razao.py` | Mede a razão de foco em vários recortes de um mosaico já montado, para verificar se a métrica é comparável entre campos. |

---

## Fluxo de uso

```
1. propriedades_camera.py   (opcional) descobrir nomes das propriedades de cor
2. calibrar_camera.py       balanço de branco + exposição  ─┐
3. calibrar_fov.py          campo de visão (µm)            ─┼─► copiar para
4. calibrar_razao_foco.py   curva razão × Z                ─┘   parametros.py

5. deteccao.py              varredura + detecção
   (ou varredura_mosaico.py para uma visão geral em mosaico)

6. montar_mosaico.py "<pasta da varredura>"    monta o mosaico
```

O `deteccao.py` pergunta, no início, qual modelo de foco usar:

- **Plano** — 3 pontos no mínimo. Rápido, serve quando a amostra é praticamente plana.
- **Superfície quadrática** — 10 pontos no mínimo. Mais precisa em magnificações altas.

Onde colocar os pontos **não é escolha sua**: o script calcula as posições que
dão o melhor condicionamento, leva o estágio até cada uma e recusa geometrias
degeneradas antes de você focar o primeiro ponto. Você só precisa focar.

---

## O que sai de cada varredura

### `deteccao.py` → `Deteccao_<data-hora>/`

| Item | Conteúdo |
|---|---|
| `mosaico/` | **Todas** as posições da grade, sem exceção. É a entrada do `montar_mosaico.py`, que precisa da grade completa. |
| `flakes/` | Só as posições que deram positivo: a imagem original e uma cópia `flakes_pos_*.tif` com os flakes marcados. Serve para conferir os achados enquanto a varredura ainda roda. |
| `coordenadas_flakes_*.csv` | Uma linha por posição com flake: nome do arquivo, X, Y, Z, a origem da grade (`X0`/`Y0`, o tile `i=0, j=0`) e a quantidade. O `X0`/`Y0` se repete em toda linha para que dê para converter em coordenadas relativas sem consultar mais nada. |
| `metricas_foco_*.csv` | Razão, textura, nitidez e o resultado do autofoco em cada posição. Gravado enquanto `REGISTRAR_METRICAS = True`. |
| `varredura.json` | Passo real, FOV, sobreposição e dimensões da grade. É o que o `montar_mosaico.py` usa para encaixar os tiles na escala certa. |

### `varredura_mosaico.py` → `Varredura_<data-hora>/`

Os `.tif` soltos na própria pasta, mais o `varredura.json`.

### `montar_mosaico.py` → dentro da pasta da varredura

`mosaico_final.tif` e `mosaico.json` (registra o `FATOR_REDUCAO` usado).

A montagem **não** acontece dentro da varredura de propósito: é a etapa mais
cara em memória do pipeline, e uma falha ali depois de horas de microscópio
jogaria tudo fora. Como os tiles já estão em disco, ela virou um script à parte,
repetível.

---

## Observações

- **Onde mexer:** valores de calibração ficam em `parametros.py`, caminhos em
  `caminhos.py`. Evite espalhar números soltos pelos scripts.
- **Duas máquinas.** O `caminhos.py` diverge de propósito entre o laptop de
  desenvolvimento e o PC do microscópio. Não tente uniformizar.
- **Sobreposição entre tiles.** `SOBREPOSICAO` faz o estágio avançar menos que
  um campo inteiro, para que um flake em cima da divisa não seja cortado em dois
  pedaços medidos separadamente. O custo é o número de imagens crescer com
  `1/(1-s)²`.
- **A borda final fica meio campo aquém.** A posição do estágio é o centro do
  campo e a malha começa no ponto marcado, então sobra meio campo *fora* da área
  na borda inicial e faltam ~35 µm na borda final. Se a borda importar, marque
  os cantos com folga. Ver `NOTAS_REVISAO.md`.
- **Pendências conhecidas** (valores a calibrar, um achado deixado de propósito,
  efeitos colaterais das últimas correções) estão em `NOTAS_REVISAO.md`.
