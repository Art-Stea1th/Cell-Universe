using System;
using System.Windows.Media;
using System.Collections.Generic;

namespace CellUniverse.Models {

    using Infrastructure.Interfaces;
    using TheGameOfLife;
    using NUniverse = TheGameOfLife.CLI.Universe;

    public sealed class Multiverse : IUniverseModelColored {

        private int width, height;

        private List<IUniverseModel> layers;
        private List<Color> colors;

        private Random random = new Random();
        private ColorWorker cworker = new ColorWorker();

        IEnumerable<Color[,]> IUniverseModelColored.NextScreen {
            get {
                Color[,] result = new Color[height, width];

                for (int i = 0; i < layers.Count; i++) {
                    foreach (var nextCell in layers[i].NextGeneration) {
                        if (result[nextCell.Item2, nextCell.Item1] == Color.FromArgb(0, 0, 0, 0)) {
                            result[nextCell.Item2, nextCell.Item1] = colors[i];
                        }
                    }
                    yield return result;
                }
            }
        }

        public Multiverse(int width, int height, int layersCount) {
            Initialize(width, height, layersCount);
        }

        private void Initialize(int width, int height, int layersCount) {

            this.width = width;
            this.height = height;

            layers = new List<IUniverseModel>(layersCount);
            for (int i = 0; i < layersCount; i++) {
                //layers.Add(new Universe(width, height));
                layers.Add(new NUniverse(width, height));
            }

            colors = cworker.GetTintsFromColor(
                Color.FromRgb((byte)random.Next(255), (byte)random.Next(255), (byte)random.Next(255)), layersCount);            
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
    }
}