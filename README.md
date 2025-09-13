Olá, seja bem vindo(a) ao meu repositório Challenge!
É um prazer compartilhar com você meu projeto, ideias, inovações e conhecimentos através desse readme.

O repositório Click-Fidelidade surgiu durante meu primeiro ano de graduação em Data Science pela FIAP. 

Todo ano a faculdade desafia seus alunos através de um problema real que uma de suas empresas parceiras compartilha com os alunos dos mais variados cursos, com o objetivo de testar suas habilidades de análise, modelagem, visualização, machine learning e entre outros. 

O tão famoso "Challenge" é uma oportunidade do aluno(a) desmonstrar suas habilidades e conhecimentos em um contexto real de mercado, fornecendo uma experiência imersiva no campo de atuação na área de dados (que é o meu caso).

A empresa parceira desse Desafio em Ciência de Dados foi a ClickBus. A ClickBus é uma plataforma brasileira de venda online de passagens de ônibus, funcionando como uma "travel tech" que conecta passageiros a diversas empresas de ônibus, permitindo a compra de bilhetes através do seu site e aplicativo móvel. A empresa oferece uma grande variedade de opções de destinos, horários e classes de serviço, além de diversas ferramentas para facilitar a compra e otimizar a experiência do usuário, como comparações de preços, seleção de assentos e pagamentos facilitados.

O Challenge foi iniciado com 3 desafios principais:
# 1 - Perfil de Compra: Segmentar clientes com base no histórico de compras para entender diferentes perfis de viajantes e direcionar estratégias de marketing. (Extra: construir um dashboard).
# 2 - Previsão da Próxima Compra: Prever se um cliente realizará uma compra nos próximos 7 ou 30 dias (classificação binária). (Extra: prever o número de dias até a próxima compra).
# 3-  Previsão do Próximo Trecho: Prever qual trecho (origem-destino) um cliente tem maior probabilidade de comprar em sua próxima viagem (classificação multi-classe ou recomendação). (Extra: combinar com o desafio 2, entregando data e trecho).

**O que foi possível concluir**: O problema central que identifiquei é a baixa retenção e a falta de lealdade na base de clientes da ClickBus. Nossas análises de segmentação revelaram que a maioria dos clientes se enquadra nos perfis "em Risco" ou "Novos/Casuais", com baixa frequência de compra. Isso indica que a empresa tem uma alta rotatividade de clientes e não está aproveitando todo o potencial de sua base.

A falta de retenção gera um impacto negativo direto no resultado da ClickBus. Um cliente não leal tem um Valor Vitalício (CLV) baixo, pois faz uma ou poucas compras. Isso aumenta o Custo de Aquisição de Cliente (CAC), pois a empresa precisa gastar mais em marketing para atrair novos compradores, em vez de investir na sua base atual. Em longo prazo, esse cenário ameaça o crescimento sustentável da empresa e a deixa vulnerável à concorrência.


**IDEIA PROPOSTA**: Um programa de fidelidade para retenção de clientes chamado "ClickBus"

**Primeira etapa ETL(Extract, Transform, Load):** Através da ferramenta Pychmarm utilizei análise RFM para segmentar o perfil de clientes afim de trazer uma melhor visão de negócio.

**Segunda etapa - Recorrência com Machine Learning:** Para incentivar a recorrência, utilizei machine learning para construir um modelo preditivo que é capaz de prever, com alta precisão, se um cliente irá comprar nos próximos 7 dias Foi definido uma data de corte (2024-01-01) para separar os dados em um período de treino  e um período de teste (compras feitas nos 7 dias seguintes). Usando as métricas de RFM, criei as características que o modelo usaria para "aprender" o comportamento de compra.
Utilizei um modelo de Regressão Logística para resolver o problema de classificação binária. O modelo foi treinado para responder "sim" (compra) ou "não" (não compra).

**Terceira etapa - Modelo de previsão:** Consegui construir um modelo de Machine Learning que consegue prever, com uma acurácia relevante, qual rota um cliente tem maior probabilidade de comprar em sua próxima viagem. Este modelo pode ser usado para personalizar ofertas e direcionar o marketing de forma mais eficiente, sugerindo o trecho correto para o cliente certo.

O script analisa a última rota comprada por um cliente e, com base em padrões históricos, prevê qual será a sua próxima rota.


Todos os scripts estão disponibilizados nesse repositório!
