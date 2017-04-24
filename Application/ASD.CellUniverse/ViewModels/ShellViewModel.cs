using System;
using System.Collections.Generic;
using System.Windows;
using System.Windows.Media;
using System.Windows.Media.Imaging;

namespace ASD.CellUniverse.ViewModels {

    using Converters;
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

        // --- TEMP >> ---

        int width = 321, height = 200;

        private WriteableBitmap pixelData;
        public WriteableBitmap PixelData {
            get => pixelData;
            set => SetProperty(ref pixelData, value);
        }

        private TempBool2dToByteArrayConverter converter = new TempBool2dToByteArrayConverter();

        // --- << TEMP ---

        public ShellViewModel() {
            
            PixelData = new WriteableBitmap(width, height, 96.0, 96.0, PixelFormats.Bgra32, null);

            generationAlgorithms = new List<IGenerationAlgorithm> { new RandomMixer(), new TheGameOfLife() };

            Generator = new FrameGenerationService(generationAlgorithms[generationAlgorithmSelectedIndex]);

            Generator.NextFrameReady += (a) => UpdatePixelData(a);
            Generator.GeneratedData = CreateRandom(321, 200);

            Controller = new ApplicationStateMachine();
            Controller.Started += Generator.Play;
            Controller.Paused += Generator.Pause;
            Controller.Resumed += Generator.Resume;
            Controller.Stopped += Generator.Stop;
            Controller.Reseted += Generator.Reset;
        }

        private void UpdatePixelData(bool[,] array)
            => PixelData.WritePixels(
                new Int32Rect(0, 0, width, height),
                converter.Convert(array, typeof(byte[]), null, null) as byte[],
                PixelData.PixelWidth * sizeof(int), 0);

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
    }
}