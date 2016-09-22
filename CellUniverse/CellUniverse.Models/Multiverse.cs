using System;
using System.Collections.Generic;
using System.Windows.Media;


namespace CellUniverse.Models {

    using Infrastructure.Interfaces;

    public sealed class Multiverse : ICellUniverse {

        private List<bool[,]> layers;
        private List<Color>   colors;
        private ICellAlgorithm algorithm;

        private Random random = new Random();

        public Multiverse(int width, int height, int layersCount, ICellAlgorithm algorithm) {
            Initialize(width, height, layersCount, algorithm);
        }

        private void Initialize(int width, int height, int layersCount, ICellAlgorithm algorithm) {
            layers = GetRandomData(width, height, layersCount);
            //colors = new ColorWorker().GetTintsFromColor(Color.FromRgb(0, 125, 168), layersCount);
            colors = new ColorWorker().GetTintsFromColor(
                Color.FromRgb((byte)random.Next(255), (byte)random.Next(255), (byte)random.Next(255)), layersCount);
            this.algorithm = algorithm;
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

        Color[,] ICellUniverse.GetNext() {

            int width  = layers[0].GetLength(1);
            int height = layers[0].GetLength(0);

            layers = GetNext(layers);
            Color[,] result = new Color[height, width];

            for (int i = 0; i < layers.Count; i++) {

                for (int y = 0; y < height; y++) {
                    for (int x = 0; x < width; x++) {
                        if (layers[i][y, x]) {
                            result[y, x] += colors[i];
                        }
                    }
                }
            }
            return result;
        }

        private List<bool[,]> GetNext(List<bool[,]> prevLayers) {

            List<bool[,]> result = new List<bool[,]>(prevLayers.Count);

            for (int i = 0; i < prevLayers.Count; i++) {
                result.Add(algorithm.NextGeneration(prevLayers[i]));
            }
            return result;
        }
    }
}