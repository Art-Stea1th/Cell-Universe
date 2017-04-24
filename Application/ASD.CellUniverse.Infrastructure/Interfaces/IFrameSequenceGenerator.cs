using System;
using System.ComponentModel;
using System.Windows.Media;

namespace ASD.CellUniverse.Infrastructure.Interfaces {

    public interface IFrameSequenceGenerator : INotifyPropertyChanged {

        event Action<bool[,]> NextFrameReady;

        double MinFPS { get; }
        double MaxFPS { get; }
        double FPS { get; set; }
        DoubleCollection FPSCollection { get; }

        IGenerationAlgorithm GenerationAlgorithm { get; set; }

        bool[,] GeneratedData { get; set; }

        void Play();
        void Pause();
        void Resume();

        void Stop();
        void Reset();
    }
}