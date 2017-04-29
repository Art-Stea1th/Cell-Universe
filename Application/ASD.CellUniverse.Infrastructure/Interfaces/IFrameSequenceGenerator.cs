using System;
using System.ComponentModel;
using System.Windows.Media;

namespace ASD.CellUniverse.Infrastructure.Interfaces {

    public interface IFrameSequenceGenerator : INotifyPropertyChanged {

        event Action<byte[,]> NextFrameReady;

        double MinFPS { get; }
        double MaxFPS { get; }
        double FPS { get; set; }
        DoubleCollection FPSCollection { get; }

        IMatrixMutator GenerationAlgorithm { get; set; }

        byte[,] GeneratedData { get; set; }

        void Play();
        void Pause();
        void Resume();

        void Stop();
        void Reset();
    }
}