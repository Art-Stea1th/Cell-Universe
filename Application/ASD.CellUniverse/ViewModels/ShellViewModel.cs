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

        // --- TEMP >> ---

        int width = 80, height = 50;

        private byte[,] intencityData;
        public byte[,] IntencityData {
            get => intencityData;
            set => SetProperty(ref intencityData, value);
        }

        // --- << TEMP ---

        public ShellViewModel() {

            IntencityData = new byte[width, height];

            generationAlgorithms = new List<IGenerationAlgorithm> { new RandomMixer(), new TheGameOfLife() };

            Generator = new FrameGenerationService(generationAlgorithms[generationAlgorithmSelectedIndex]);

            Generator.NextFrameReady += (a) => UpdateIntencityData(a);
            Generator.GeneratedData = CreateRandom(width, height);

            Controller = new ApplicationStateMachine();
            Controller.Started += Generator.Play;
            Controller.Paused += Generator.Pause;
            Controller.Resumed += Generator.Resume;
            Controller.Stopped += Generator.Stop;
            Controller.Reseted += Generator.Reset;

            // TMP
            Controller.Stopped += () => Generator.GeneratedData = CreateRandom(width, height);
            Controller.Reseted += () => Generator.GeneratedData = CreateRandom(width, height);

        }

        private void UpdateIntencityData(byte[,] newIntencityData)
            => IntencityData = newIntencityData;

        // TMP
        private byte[,] CreateRandom(int width, int height) {

            var random = new Random();
            var result = new byte[width, height];

            for (var y = 0; y < height; y++) {
                for (var x = 0; x < width; x++) {
                    result[x, y] = random.Next() % 2 == 1 ? (byte)255 : (byte)0;
                }
            }
            return result;
        }
    }
}