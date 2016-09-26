using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Media;

namespace CellUniverse.Models {

    using Infrastructure.Interfaces;
    using Algorithms;

    public sealed class Multiverse : ICellUniverse {

        private short width, height;

        private List<ICellAlgorithm> layers;
        private List<Color> colors;

        private Random random = new Random();
        private ColorWorker cworker = new ColorWorker();        

        public Multiverse(int width, int height, int layersCount) {
            Initialize(width, height, layersCount);
        }

        private void Initialize(int width, int height, int layersCount) {

            this.width = (short)width;
            this.height = (short)height;

            GenerateNotIdenticalLayers(this.width, this.height, (byte)layersCount);

            colors = cworker.GetTintsFromColor(
                Color.FromRgb((byte)random.Next(255), (byte)random.Next(255), (byte)random.Next(255)), layersCount);
        }

        private void GenerateNotIdenticalLayers(short width, short height, byte layersCount) {

            layers = new List<ICellAlgorithm>(layersCount);

            for (short i = 0; i < layersCount; i++) {

                bool[,] newLayer = GetRandomLayer(width, height);
                bool IdenticalGeneration = true;

                while (IdenticalGeneration && i > 1) {
                    newLayer = GetRandomLayer(width, height);
                    foreach (var layer in layers) {
                        IdenticalGeneration = layer.IsIdentical(newLayer);
                    }
                }
                layers.Add(new TheGameOfLife(newLayer));
            }
        }

        private bool[,] GetRandomLayer(short width, short height) {
            bool[,] layer = new bool[width, height];
            for (int x = 0; x < width; x++) {
                for (int y = 0; y < height; y++) {
                    layer[x, y] = random.Next(2) == 1;
                }
            }
            return layer;
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