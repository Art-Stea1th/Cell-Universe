using System.ComponentModel;
using System.Windows.Input;
using System.Windows.Media;

namespace ASD.CellUniverse.Infrastructure.Interfaces {

    public interface IApplicationFacade : INotifyPropertyChanged {

        uint[,] Matrix { get; }

        int GenerationWidth { get; set; }
        int GenerationHeight { get; set; }
        bool MatrixReadyToChange { get; }
        bool MatrixReadyToMutate { get; }

        ISeedGenerator SeedWriter { get; set; }
        ICommand WriteSeed { get; }
        IMutationAlgorithm Algorithm { get; set; }

        DoubleCollection MPSCollection { get; }
        double MinMPS { get; }
        double MaxMPS { get; }
        double MPS { get; set; }

        State State { get; }
        ICommand Start { get; }
        ICommand Stop { get; }
    }
}