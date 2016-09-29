using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Media;

namespace CellUniverse.Models {

    using Infrastructure.Interfaces;
    using Algorithms;

    public sealed class Multiverse : ICellUniverse {

        private int width, height;

        private List<ICellAlgorithm> layers;
        private List<Color> colors;

        private Random random = new Random();
        private ColorWorker cworker = new ColorWorker();        

        public Multiverse(int width, int height, int layersCount) {
            Initialize(width, height, layersCount);
        }

        private void Initialize(int width, int height, int layersCount) {

            this.width = width;
            this.height = height;

            GenerateNotIdenticalLayers(this.width, this.height, (byte)layersCount);

            colors = cworker.GetTintsFromColor(
                Color.FromRgb((byte)random.Next(255), (byte)random.Next(255), (byte)random.Next(255)), layersCount);
        }

        private void GenerateNotIdenticalLayers(int width, int height, byte layersCount) {

            //List<bool[,]> layersData = new List<bool[,]>(layersCount);

            //for (int i = 0; i < layersCount; i++) {

            //    bool[,] newLayer = GetRandomLayer(width, height);
            //    bool IdenticalGeneration = true;

            //    while (IdenticalGeneration && i > 1) {
            //        newLayer = GetRandomLayer(width, height);
            //        foreach (var layer in layersData) {
            //            IdenticalGeneration = IsIdentical(layer, newLayer);
            //        }
            //    }
            //    layersData.Add(newLayer);
            //}
            layers = new List<ICellAlgorithm>(layersCount);
            for (int i = 0; i < layersCount; i++) {
                //layers.Add(new TheGameOfLife(layersData[i]));
                layers.Add(new TheGameOfLifeNativeWrapper(width, height));
            }
        }

        private bool[,] GetRandomLayer(int width, int height) {
            bool[,] layer = new bool[width, height];
            for (int x = 0; x < width; x++) {
                for (int y = 0; y < height; y++) {
                    layer[x, y] = random.Next(2) == 1;
                }
            }
            return layer;
        }

        public bool IsIdentical(bool[,] layerA, bool[,] layerB) {
            for (int x = 0; x < layerA.GetLength(0); x++) {
                for (int y = 0; y < layerA.GetLength(1); y++) {
                    if (layerA[x, y] != layerB[x, y]) {
                        return false;
                    }
                }
            }
            return true;
        }

        IEnumerable<Color[,]> ICellUniverse.GetNext() {

            Color[,] result = new Color[height, width];

            for (int i = 0; i < layers.Count; i++) {
                foreach (var nextCell in layers[i].NextGeneration()) {
                    if (result[nextCell.Item2, nextCell.Item1] == Color.FromArgb(0, 0, 0, 0)) {
                        result[nextCell.Item2, nextCell.Item1] = colors[i];
                    }
                }
                yield return result;
            }
        }
    }
}