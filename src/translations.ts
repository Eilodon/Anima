
import { BreathingType } from './types';

export type Language = 'en' | 'vi';

export const TRANSLATIONS = {
  en: {
    phases: {
      inhale: 'INHALE',
      exhale: 'EXHALE',
      holdIn: 'HOLD',
      holdOut: 'STILL',
      hold: 'HOLD'
    },
    shortPhases: {
      in: 'IN',
      out: 'OUT',
      hold: 'HOLD'
    },
    ui: {
      title: 'ZENB',
      focusMode: 'Focus',
      paused: 'PAUSED',
      resume: 'Resume',
      pause: 'Pause',
      end: 'End Session',
      begin: 'Start Breathing',
      cycle: 'Cycle',
      selectRhythm: 'Select your path',
      settings: 'Settings',
      finish: 'Finish',
      timeBreathed: 'Time in Flow',
      cycles: 'Cycles Completed',
      sessionComplete: 'Session Complete',
      mindClear: 'Carry this stillness with you.',
      welcome: 'Welcome to ZenB',
      welcomeDesc: 'A sanctuary for your mind.',
      findRhythm: 'Find Your Rhythm',
      findRhythmDesc: 'Patterns designed to align your state of being.',
      breatheLight: 'Breathe with Light',
      breatheLightDesc: 'Expand with the light. Release as it fades.',
      continue: 'Continue',
      beginJourney: 'Begin Journey',
      streak: 'Mindful Streak',
      dayStreak: 'Day Streak',
    },
    history: {
      title: 'Journal',
      totalSessions: 'Sessions',
      totalMinutes: 'Minutes of Zen',
      noHistory: 'The journey of a thousand miles begins with a single breath.',
      clear: 'Reset History',
      today: 'Today',
      min: 'm',
      sec: 's'
    },
    settings: {
      header: 'Preferences',
      immersion: 'IMMERSION',
      sounds: 'Audio',
      haptics: 'Haptics',
      visuals: 'VISUALS',
      graphics: 'Graphics',
      reduceMotion: 'Reduce Motion',
      showTimer: 'Show Progress',
      language: 'Language',
      soundPack: 'Soundscape',
      quality: {
        auto: 'Auto',
        low: 'Eco',
        medium: 'Balanced',
        high: 'High'
      },
      hapticStrength: {
        light: 'Subtle',
        medium: 'Balanced',
        heavy: 'Strong'
      },
      soundPacks: {
        musical: 'Zen Melodies',
        bells: 'Tibetan Bowl',
        breath: 'Nature',
        'real-zen': 'Real Breath (Studio)',
        'voice-en': 'Voice (English)',
        'voice-vi': 'Voice (Vietnamese)',
        'voice-12': 'Counting (1-2)'
      }
    },
    patterns: {
      '4-7-8': {
        label: 'Tranquility',
        tag: 'Sleep & Anxiety',
        description: 'A natural tranquilizer. Calms the nervous system.'
      },
      'box': {
        label: 'Focus',
        tag: 'Concentration',
        description: 'Heightens performance and concentration.'
      },
      'calm': {
        label: 'Balance',
        tag: 'Coherence',
        description: 'Restores natural heart rate variability.'
      },
      'coherence': {
        label: 'Resonance',
        tag: 'Heart Health',
        description: 'Synchronizes heart rate and breathing (6 breaths/min).'
      },
      'deep-relax': {
        label: 'Deep Rest',
        tag: 'Stress Relief',
        description: 'Extended exhalation triggers deep relaxation.'
      },
      '7-11': {
        label: '7-11',
        tag: 'Panic Relief',
        description: 'Powerful technique to stop panic attacks instantly.'
      },
      'awake': {
        label: 'Energize',
        tag: 'Wake Up',
        description: 'Rapid cycling to boost oxygen and alertness.'
      },
      'triangle': {
        label: 'Triangle',
        tag: 'Yoga',
        description: 'Geometric flow for emotional stability.'
      },
      'tactical': {
        label: 'Tactical',
        tag: 'High Stress',
        description: 'Advanced Box breathing for maximum control.'
      },
      'buteyko': {
        label: 'Light Air',
        tag: 'Nasal Health',
        description: 'Gentle, reduced breathing to improve oxygen uptake.'
      },
      'wim-hof': {
        label: 'Tummo Power',
        tag: 'Immunity',
        description: 'Charge the body. Inhale deeply, let go. Repeat.'
      }
    }
  },
  vi: {
    phases: {
      inhale: 'HÍT VÀO',
      exhale: 'THỞ RA',
      holdIn: 'GIỮ',
      holdOut: 'GIỮ',
      hold: 'GIỮ'
    },
    shortPhases: {
      in: 'HÍT',
      out: 'THỞ',
      hold: 'GIỮ'
    },
    ui: {
      title: 'ZENB',
      focusMode: 'Chánh niệm',
      paused: 'TẠM DỪNG',
      resume: 'Tiếp tục',
      pause: 'Dừng lại',
      end: 'Kết thúc',
      begin: 'Bắt đầu',
      cycle: 'Vòng',
      selectRhythm: 'Chọn liệu pháp',
      settings: 'Cài đặt',
      finish: 'Hoàn tất',
      timeBreathed: 'Thời gian tịnh',
      cycles: 'Số vòng thở',
      sessionComplete: 'Hoàn thành',
      mindClear: 'Mang theo sự bình an này vào từng khoảnh khắc.',
      welcome: 'Chào đón đến ZenB',
      welcomeDesc: 'Không gian tĩnh lặng cho tâm hồn bạn.',
      findRhythm: 'Tìm nhịp điệu',
      findRhythmDesc: 'Chọn bài tập phù hợp với tâm ý hiện tại.',
      breatheLight: 'Thở cùng ánh sáng',
      breatheLightDesc: 'Hít vào khi ánh sáng lan tỏa. Buông thư khi ánh sáng thu về.',
      continue: 'Tiếp bước',
      beginJourney: 'Bắt đầu hành trình',
      streak: 'Tinh tấn',
      dayStreak: 'Ngày liên tục',
    },
    history: {
      title: 'Hành trình',
      totalSessions: 'Phiên tập',
      totalMinutes: 'Phút bình an',
      noHistory: 'Hành trình vạn dặm bắt đầu từ một hơi thở.',
      clear: 'Xóa dữ liệu',
      today: 'Hôm nay',
      min: 'phút',
      sec: 'giây'
    },
    settings: {
      header: 'Tùy chỉnh',
      immersion: 'TRẢI NGHIỆM',
      sounds: 'Âm thanh',
      haptics: 'Xúc giác',
      visuals: 'HÌNH ẢNH',
      graphics: 'Đồ họa',
      reduceMotion: 'Giảm chuyển động',
      showTimer: 'Vòng tiến độ',
      language: 'Ngôn ngữ',
      soundPack: 'Âm dẫn',
      quality: {
        auto: 'Tự động',
        low: 'Tiết kiệm',
        medium: 'Cân bằng',
        high: 'Cao'
      },
      hapticStrength: {
        light: 'Nhẹ',
        medium: 'Vừa',
        heavy: 'Mạnh'
      },
      soundPacks: {
        musical: 'Giai điệu Zen',
        bells: 'Chuông xoay',
        breath: 'Hơi thở tự nhiên',
        'real-zen': 'Hơi thở thật (Studio)',
        'voice-en': 'Giọng đọc (Tiếng Anh)',
        'voice-vi': 'Giọng đọc (Tiếng Việt)',
        'voice-12': 'Đếm nhịp (1-2)'
      }
    },
    patterns: {
      '4-7-8': {
        label: 'An Yên',
        tag: 'Ngủ & Lo âu',
        description: 'Liều thuốc tự nhiên xoa dịu hệ thần kinh.'
      },
      'box': {
        label: 'Định Tâm',
        tag: 'Tập trung',
        description: 'Gia tăng sự tỉnh thức và hiệu suất trí tuệ.'
      },
      'calm': {
        label: 'Cân Bằng',
        tag: 'Điều hòa',
        description: 'Khôi phục nhịp điệu tự nhiên của trái tim.'
      },
      'coherence': {
        label: 'Cộng Hưởng',
        tag: 'Tim mạch',
        description: 'Đồng bộ nhịp tim và hơi thở (tỉ lệ vàng).'
      },
      'deep-relax': {
        label: 'Thư Giãn Sâu',
        tag: 'Giảm Stress',
        description: 'Kéo dài hơi thở ra để kích hoạt sự thư thái.'
      },
      '7-11': {
        label: '7-11',
        tag: 'Cấp Cứu',
        description: 'Kỹ thuật mạnh mẽ để cắt cơn hoảng loạn tức thì.'
      },
      'awake': {
        label: 'Tỉnh Thức',
        tag: 'Năng lượng',
        description: 'Nhịp thở nhanh giúp nạp oxy và đánh thức tâm trí.'
      },
      'triangle': {
        label: 'Tam Giác',
        tag: 'Yoga',
        description: 'Mô hình hình học giúp ổn định cảm xúc.'
      },
      'tactical': {
        label: 'Chiến Thuật',
        tag: 'Áp lực cao',
        description: 'Biến thể nâng cao của Box Breathing cho tình huống căng thẳng.'
      },
      'buteyko': {
        label: 'Khí Nhẹ',
        tag: 'Sức khỏe mũi',
        description: 'Thở nhẹ và nông để tăng khả năng hấp thụ oxy.'
      },
      'wim-hof': {
        label: 'Tummo Power',
        tag: 'Miễn dịch',
        description: 'Kích hoạt năng lượng cơ thể. Hít sâu, buông lỏng. Lặp lại.'
      }
    }
  }
};
