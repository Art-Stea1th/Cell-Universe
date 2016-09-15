using System;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Collections.Generic;


namespace GameOfLife.ViewModels {

    public class ControlsPanelViewModel : ViewModelBase {

        private int _currentSpeed;


        public int MinSpeed { get; } = 1;
        public int MaxSpeed { get; } = 100;

        public int Speed {
            get {
                return _currentSpeed = IntLimiter(_currentSpeed, MinSpeed, MaxSpeed);
            }
            set {
                _currentSpeed = IntLimiter(value, MinSpeed, MaxSpeed);
                OnPropertyChanged(GetMemberName((ControlsPanelViewModel c) => c.Speed));
            }
        }

        private int IntLimiter(int value, int minValue, int maxValue) {
            value = value < minValue ? minValue : value;
            value = value > maxValue ? maxValue : value;
            return value;
        }
    }
}