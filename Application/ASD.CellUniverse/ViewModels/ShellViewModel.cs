using System;
using System.Collections.Generic;

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

        //int width = 800, height = 500; // 16 : 10
        //int width = 480, height = 300; // 16 : 10
        //int width = 400, height = 250; // 16 : 10
        int width = 320, height = 200; // 16 : 10
        //int width = 160, height = 100; // 16 : 10
        //int width = 80, height = 50; // 16 : 10
        //int width = 40, height = 25; // 16 : 10
        //int width = 32, height = 20; // 16 : 10
        //int width = 16, height = 10; // 16 : 10
        //int width = 8, height = 5; // 16 : 10


        private byte[,] intencityData;
        public byte[,] IntencityData {
            get => intencityData;
            set => SetProperty(ref intencityData, value);
        }

        // --- << TEMP ---

        public ShellViewModel() {

            IntencityData = new byte[width, height];

            generationAlgorithms = new List<IGenerationAlgorithm> { new TheGameOfLife(), new RandomMixer() };

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