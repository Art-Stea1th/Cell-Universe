using System;
using System.Collections.Generic;
using System.Windows;
using System.Windows.Media;
using System.Windows.Media.Imaging;

namespace ASD.CellUniverse.ViewModels {

    using Infrastructure.Algorithms;
    using Infrastructure.Controllers;
    using Infrastructure.Interfaces;
    using Infrastructure.MVVM;
    using Infrastructure.Services;

    public sealed class ShellViewModel : BindableBase {

        private int generationAlgorithmSelectedIndex = 0;
        public int GenerationAlgorithmSelectedIndex {
            get => generationAlgorithmSelectedIndex;
            set {
                SetProperty(ref generationAlgorithmSelectedIndex, value);
                Generator.GenerationAlgorithm = generationAlgorithms[generationAlgorithmSelectedIndex];
            }
        }
        public IEnumerable<IGenerationAlgorithm> GenerationAlgorithms => generationAlgorithms;
        private List<IGenerationAlgorithm> generationAlgorithms;

        public IFrameSequenceGenerator Generator { get; private set; }

        public IMainController Controller { get; private set; }




        private WriteableBitmap pixelData;
        public WriteableBitmap PixelData {
            get => pixelData;
            set => SetProperty(ref pixelData, value);
        }

        public ShellViewModel() {

            generationAlgorithms = new List<IGenerationAlgorithm> { new RandomMixer(), new TheGameOfLife() };

            Generator = new FrameGenerationService(generationAlgorithms[generationAlgorithmSelectedIndex]);

            Generator.NextFrameReady += (a) => UpdatePixelData(a, Colors.DarkGray, Colors.Transparent);
            Generator.GeneratedData = CreateRandom(321, 200);

            Controller = new ApplicationStateMachine();
            Controller.Started += Generator.Start;
            Controller.Paused += Generator.Pause;
            Controller.Stopped += Generator.Stop;
        }

        private void UpdatePixelData(bool[,] array, Color trueColor, Color falseColor)
            => PixelData = NewBitmapFrom(array, trueColor, falseColor);

        private bool[,] CreateRandom(int width, int height) {

            var random = new Random();
            var result = new bool[width, height];

            for (var y = 0; y < height; y++) {
                for (var x = 0; x < width; x++) {
                    result[x, y] = random.Next() % 2 == 1;
                }
            }
            return result;
        }

        private WriteableBitmap NewBitmapFrom(bool[,] array, Color trueColor, Color falseColor) {
            var width = array.GetLength(0);
            var height = array.GetLength(1);

            var colors = new Color[width, height];

            for (var y = 0; y < height; y++) {
                for (var x = 0; x < width; x++) {
                    colors[x, y] = array[x, y] ? trueColor : falseColor;
                }
            }
            return NewBitmapFrom(colors);
        }

        private WriteableBitmap NewBitmapFrom(Color[,] colors) {

            int width = colors.GetLength(0), height = colors.GetLength(1);
            var bytes = new byte[width * height * sizeof(int) / sizeof(byte)];

            for (var y = 0; y < height; ++y) {
                for (var x = 0; x < width; ++x) {
                    SetColorToByteArray(bytes, GetLinearIndex(x, y, width) * sizeof(int), colors[x, y]);
                }
            }

            var result = new WriteableBitmap(width, height, 96.0, 96.0, PixelFormats.Bgra32, null);
            result.WritePixels(new Int32Rect(0, 0, width, height), bytes, result.PixelWidth * sizeof(int), 0);

            return result;
        }

        private int GetLinearIndex(int x, int y, int width) {
            return width * y + x;
        }

        private void SetColorToByteArray(byte[] viewportByteArray, int startIndex, Color color) {
            viewportByteArray[startIndex] = color.B;
            viewportByteArray[++startIndex] = color.G;
            viewportByteArray[++startIndex] = color.R;
            viewportByteArray[++startIndex] = color.A;
        }
    }
}