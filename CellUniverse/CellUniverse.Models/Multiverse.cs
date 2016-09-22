using System;
using System.Collections.Generic;
using System.Windows.Media;


namespace CellUniverse.Models {

    using Infrastructure.Interfaces;

    public sealed class Multiverse : ICellUniverse {

        private List<bool[,]> layers;
        private List<Color>  colors;

        private Random random;

        public Multiverse(int width, int height, int layersCount) {

            int hardcodeLayersCount = 3; // !!! temp impl.

            Initialize(width, height, hardcodeLayersCount);
        }

        private void Initialize(int width, int height, int layersCount) {
            random = new Random();
            layers = GetRandomData(width, height, layersCount);
            colors = new ColorWorker().GetColorList(layersCount);
        }

        private List<bool[,]> GetRandomData(int width, int height, int layersCount) {

            var result = new List<bool[,]>(layersCount);

            for (int i = 0; i < layersCount; i++) {
                result.Add(new bool[height, width]);

                for (int y = 0; y < height; y++) {
                    for (int x = 0; x < width; x++) {
                        result[i][y, x] = random.Next(2) == 1;
                    }
                }
            }
            return result;
        }

        public Color[,] GetNext() {

            GenerateNext();

            int width  = layers[0].GetLength(1);
            int height = layers[0].GetLength(0);

            Color[,] result = new Color[height, width];

            for (int i = 0; i < layers.Count; i++) {

                for (int y = 0; y < height; y++) {
                    for (int x = 0; x < width; x++) {
                        //result[y, x] += Color.FromRgb(16, 16, 16);
                        if (layers[i][y, x]) {
                            result[y, x] += colors[i];
                        }
                    }
                }
            }
            return result;
        }

        private void GenerateNext() {
            layers = GetNext(layers);
        }

        private List<bool[,]> GetNext(List<bool[,]> prevLayers) {

            List<bool[,]> result = new List<bool[,]>(prevLayers.Count);

            for (int i = 0; i < prevLayers.Count; i++) {
                result.Add(NextGenerationFromSource(prevLayers[i]));
            }
            return result;
        }

        private bool[,] NextGenerationFromSource(bool[,] source) {

            var nextGeneration = new bool[source.GetLength(0), source.GetLength(1)];

            for (int y = 0; y < source.GetLength(0); y++) {
                for (int x = 0; x < source.GetLength(1); x++) {

                    int neighboursCount = CountNeighbours(source, x, y);

                    if ((neighboursCount == 2 || neighboursCount == 3) && IsAlive(source, x, y))
                        nextGeneration[y, x] = true;

                    if ((neighboursCount < 2 || neighboursCount > 3) && IsAlive(source, x, y))
                        nextGeneration[y, x] = false;

                    if (neighboursCount == 3 && !IsAlive(source, x, y))
                        nextGeneration[y, x] = true;
                }
            }
            return nextGeneration;
        }

        private int CountNeighbours(bool[,] generation, int posX, int posY) {

            int counter = 0;

            for (int y = posY - 1; y < posY + 2; y++) {
                for (int x = posX - 1; x < posX + 2; x++) {
                    if (x == posX && y == posY)
                        continue;

                    int py = y, px = x;

                    if (py == -1) py = generation.GetLength(0) - 1;
                    else if (py == generation.GetLength(0)) py = 0;

                    if (px == -1) px = generation.GetLength(1) - 1;
                    else if (px == generation.GetLength(1)) px = 0;

                    if (IsAlive(generation, px, py))
                        counter++;
                }
            }
            return counter;
        }

        private bool IsAlive(bool[,] generation, int posX, int posY) {
            return generation[posY, posX] ? true : false;
        }        
    }
}